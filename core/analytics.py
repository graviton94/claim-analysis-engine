"""
core/analytics.py
=================
Advanced Analytics Module for Food Safety Risk Scoring.
Implementation Level: Phase 3.1 (Statistical Stability & Safety Guards)
 - Config Refactoring
 - Small Sample Variance Guard
 - Early Month Velocity Guard
 - Conditional Safe Zone
 - [New] Phase 2.5: Zero-Filling Data Preparation
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
from typing import Dict, Tuple, List, Union, Optional
import statsmodels.api as sm
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass

# --- 1. Configuration Management (Phase 3.1) ---
@dataclass
class RiskConfig:
    # Data Requirements
    MIN_DATA_POINTS: int = 3
    MIN_SAMPLE_FOR_NB: int = 6  # 음이항분포 사용을 위한 최소 표본 수
    
    # Statistical Parameters
    EWMA_LAMBDA: float = 0.2
    NELSON_WINDOW_BIAS: int = 9
    NELSON_WINDOW_TREND: int = 6
    STL_MIN_PERIODS: int = 24
    
    # Scoring Weights
    SCORE_ACCIDENT: int = 100
    SCORE_CRITICAL_BONUS: int = 10
    
    SCORE_NELSON_DEV: int = 30   # 관리이탈
    SCORE_NELSON_BIAS: int = 20  # 평균이동
    SCORE_NELSON_TREND: int = 20 # 지속상승
    SCORE_MOMENTUM: int = 15     # 연속상승(3M)
    
    SCORE_PARTIAL_ABS: int = 30  # 조기과다
    SCORE_PARTIAL_VEL: int = 40  # 속도급증
    SCORE_PARTIAL_WARN: int = 30 # 속도주의
    
    # Thresholds
    THRESHOLD_RED_CRIT: int = 75
    THRESHOLD_RED_GEN: int = 85
    THRESHOLD_YEL_CRIT: int = 50
    THRESHOLD_YEL_GEN: int = 60
    
    # Safety Guards
    MIN_PROGRESS_FOR_VELOCITY: float = 0.2  # 월 진행률 20% 이상일 때만 속도 판정

# Critical Grades Definition
CRITICAL_GRADES_SET = {'중대', '위험', '사고'}

class RiskScoringEngine:
    def __init__(self, data_series: pd.Series, grade: str = None, target_month_str: str = None):
        self.series = data_series.sort_index()
        self.grade = grade
        self.is_critical = (grade in CRITICAL_GRADES_SET) if grade else False
        self.cfg = RiskConfig()
        
        # 현재 월(Partial Month) 여부 확인
        self.is_partial_month = False
        self.progress_ratio = 1.0
        
        if target_month_str:
            try:
                today = datetime.now()
                target_date = datetime.strptime(target_month_str, "%Y-%m")
                
                # 분석 대상이 '이번 달'이고, 아직 달이 안 끝났다면
                if (target_date.year == today.year) and (target_date.month == today.month):
                    self.is_partial_month = True
                    # 진행률 계산
                    day_of_month = max(1, today.day)
                    days_in_month = (target_date.replace(month=target_date.month % 12 + 1, day=1) - pd.Timedelta(days=1)).day
                    self.progress_ratio = day_of_month / days_in_month
            except:
                pass 

        if len(self.series) > 0:
            self.current_value = self.series.iloc[-1]
            self.current_date = self.series.index[-1]
            self.history = self.series.iloc[:-1]
        else:
            self.current_value = 0
            self.history = pd.Series(dtype=float)
            
        self.n_obs = len(self.history)
        self.mean = self.history.mean() if self.n_obs > 0 else 0.0
        self.std = self.history.std() if self.n_obs > 1 else 0.0
        self.var = self.history.var() if self.n_obs > 1 else 0.0
        
        # [Regime] 희소성 판단
        zero_ratio = (self.history == 0).sum() / self.n_obs if self.n_obs > 0 else 0
        self.is_sparse = (self.mean < 1.0) or (zero_ratio > 0.5)

    def _calculate_sparse_score(self) -> Tuple[float, str]:
        """ [Track A] 희소 데이터 스코어링 (Phase 3.1: 소표본 가드 추가) """
        if self.mean == 0:
            raw_score = 80.0 if self.current_value > 0 else 0.0
            method = "희소유형 돌발 발생"
        else:
            # [Phase 3.1] Small Sample Variance Guard
            # 표본이 적을 때(N<6) 분산 추정은 매우 불안정하므로 보수적인 Poisson 강제
            use_nbinom = False
            if self.n_obs >= self.cfg.MIN_SAMPLE_FOR_NB:
                # 과대산포 검정 (분산이 평균의 1.2배 초과)
                if self.var > (1.2 * self.mean):
                    use_nbinom = True
            
            if use_nbinom:
                p_est = self.mean / self.var
                r_est = (self.mean * p_est) / (1 - p_est)
                p_val = 1 - nbinom.cdf(self.current_value - 1, n=r_est, p=p_est)
                method = "분포 이탈"
            else:
                p_val = 1 - poisson.cdf(self.current_value - 1, mu=self.mean)
                method = "분포 이탈"

            if p_val < 1e-5: raw_score = 100.0
            else:
                raw_score = -np.log10(p_val) * 25
                raw_score = min(100.0, max(0.0, raw_score))

        return raw_score, method

    def _calculate_momentum_score(self) -> float:
        """ [Phase 3.0] 연속 상승 모멘텀 """
        if self.n_obs < 2: return 0.0
        
        val_t = self.current_value
        val_t_1 = self.history.iloc[-1]
        val_t_2 = self.history.iloc[-2]
        
        if (val_t > val_t_1) and (val_t_1 > val_t_2):
            if val_t >= 3: 
                return self.cfg.SCORE_MOMENTUM
        return 0.0

    def _get_z_score_with_stl(self) -> Tuple[float, str]:
        """ [Track B] Z-Score (STL or Standard) """
        if self.n_obs >= self.cfg.STL_MIN_PERIODS and self.std > 0:
            try:
                decomposition = sm.tsa.seasonal_decompose(self.history, model='additive', period=12)
                residuals = decomposition.resid.dropna()
                resid_mean = residuals.mean()
                resid_std = residuals.std()
                
                last_trend = decomposition.trend.dropna().iloc[-1]
                seasonal_comp = decomposition.seasonal
                target_month_idx = self.current_date.month
                current_seasonal = seasonal_comp[seasonal_comp.index.month == target_month_idx].mean()
                
                expected_val = last_trend + current_seasonal
                current_resid = self.current_value - expected_val
                
                z = (current_resid - resid_mean) / (resid_std + 1e-6)
                return z, "정상패턴 이탈"
            except:
                pass
        
        z = (self.current_value - self.mean) / (self.std + 1e-6)
        return z, "평균 대비 급증"

    def _apply_nelson_rules(self, z_score: float) -> Tuple[float, List[str]]:
        triggered = []
        score_add = 0.0
        full = self.series
        
        # Adaptive Threshold (Phase 3.0)
        cv = (self.std / self.mean) if self.mean > 0 else 0
        base_limit = 2.5 if self.is_critical else 3.0
        
        if cv < 0.1 and self.mean > 1.0:
            limit_z = 2.0
            adaptive_msg = "(안정공정)"
        elif cv > 0.5:
            limit_z = 3.5
            adaptive_msg = "(불안정공정)"
        else:
            limit_z = base_limit
            adaptive_msg = ""
            
        warn_z = max(1.5, limit_z - 1.0)
        
        # Rule 1: Limit Violation
        if abs(z_score) > limit_z:
            triggered.append(f"정상범위 이탈")
            score_add += self.cfg.SCORE_NELSON_DEV
        elif abs(z_score) > warn_z:
            score_add += (self.cfg.SCORE_NELSON_DEV / 2)
            
        # Rule 2: Bias (Shift)
        if len(full) >= self.cfg.NELSON_WINDOW_BIAS:
            last_n = full.iloc[-self.cfg.NELSON_WINDOW_BIAS:]
            if (last_n > self.mean).all():
                triggered.append(f"지속적 상승 - {self.cfg.NELSON_WINDOW_BIAS}개월")
                score_add += self.cfg.SCORE_NELSON_BIAS
        
        # Rule 3: Trend
        if len(full) >= self.cfg.NELSON_WINDOW_TREND:
            last_n = full.iloc[-self.cfg.NELSON_WINDOW_TREND:]
            diffs = last_n.diff().dropna()
            if (diffs > 0).all():
                triggered.append(f"지속적 상승 - ({self.cfg.NELSON_WINDOW_TREND}개월)")
                score_add += self.cfg.SCORE_NELSON_TREND
                
        return score_add, triggered

    def _calculate_velocity_score(self) -> float:
            """ [Track B] 급격한 기울기 변화 감지 (일반 등급 미탐 방지) """
            if self.n_obs < 1: return 0.0
            
            prev = self.history.iloc[-1]
            # 전월 0건이거나 당월 절대값이 작으면 패스
            if prev == 0 and self.current_value < 3: return 0.0
            if self.current_value < 5: return 0.0 # 최소 5건 이상일 때만 속도 판정
            
            denom = prev if prev > 0 else 0.5 # 0으로 나누기 방지
            ratio = self.current_value / denom
            
            if ratio >= 3.0: return 30.0
            elif ratio >= 2.0: return 15.0
            return 0.0

    def calculate_score(self) -> Dict:
        # 0. 발생 없음
        if self.current_value == 0:
            return {"score": 0, "status": "🟢", "reason": "발생 없음"}

        # 1. 월중 조기 경보 (Partial Month Logic)
        partial_month_penalty = 0
        partial_reason = ""
        
        # 조기과다: 2건 이상일 때만 판정
        if self.is_partial_month and self.current_value > 1:
            # A. 절대 속도 위반
            if self.current_value >= self.mean and self.mean > 0 and self.progress_ratio < 0.7:
                partial_month_penalty = self.cfg.SCORE_PARTIAL_ABS
                partial_reason = f"상승세 가속"
            
            # B. 상대 속도 위반 (Velocity Surge) - [Phase 3.1] Safety Guard 적용
            # 월 초반(10% 미만)에는 우연에 의한 배수 뻥튀기가 심하므로 속도 판정 스킵
            elif self.progress_ratio >= self.cfg.MIN_PROGRESS_FOR_VELOCITY:
                expected_current = self.mean * self.progress_ratio
                if expected_current > 0.5 and self.current_value > (expected_current * 4.0):
                    partial_month_penalty = self.cfg.SCORE_PARTIAL_VEL
                    partial_reason = f"상승세 가속"
                elif expected_current > 0.5 and self.current_value > (expected_current * 2.5):
                    if self.is_critical:
                        partial_month_penalty = self.cfg.SCORE_PARTIAL_WARN
                        partial_reason = "상승세 가속"
        # 2. Main Scoring Variables
        total_score = 0.0
        method_str = ""
        triggered_rules = [] # Nelson Rules Trigger List
        z_score_val = 0.0    # For Safe Zone check

        # 3. Data Scarcity Check (Init)
        if self.n_obs < self.cfg.MIN_DATA_POINTS:
            # 초기 데이터 부족 시 Rule-based 처리
            if self.is_critical:
                if self.current_value >= 2: 
                     return {"score": 100, "status": "🔴", "reason": f"초기급증({partial_reason})"}
            else:
                if self.current_value >= 3:
                     return {"score": 50, "status": "🟡", "reason": f"초기주의({partial_reason})"}
                else:
                     return {"score": 0, "status": "⚪", "reason": "데이터 부족"}

        # 4. Calculation (Dense vs Sparse)
        if self.is_sparse:
            prob_score, method_str = self._calculate_sparse_score()
            
            # Sparse Trend Check
            trend_score = 0
            prev_val = self.history.iloc[-1] if len(self.history) > 0 else 0
            if prev_val > 0 and (self.current_value / prev_val) >= 3.0 and self.current_value >= 3:
                trend_score = 20
            
            total_score = prob_score + trend_score
        else: # Dense
            z_score_val, z_method = self._get_z_score_with_stl()
            method_str = z_method
            
            start_sigma = 0.5 if self.is_critical else 1.0
            base_score = min(50, max(0, (z_score_val - start_sigma) * (50 / 2.0)))
            
            # Apply Nelson Rules
            nelson_score, triggered_rules = self._apply_nelson_rules(z_score_val)
            
            # EWMA Score (Simple Moving Average Deviation)
            ewma = self.series.ewm(alpha=self.cfg.EWMA_LAMBDA, adjust=False).mean()
            z_ewma = (ewma.iloc[-1] - self.mean) / (self.std * np.sqrt(self.cfg.EWMA_LAMBDA/(2-self.cfg.EWMA_LAMBDA)) + 1e-6)
            ewma_score = 15.0 if abs(z_ewma) > 3.0 else 0.0
            
            # Velocity Score
            velocity_score = self._calculate_velocity_score()
            
            total_score = base_score + nelson_score + ewma_score + velocity_score

        # Momentum Score
        momentum_score = self._calculate_momentum_score()
        if momentum_score > 0:
            triggered_rules.append("연속 상승 모멘텀")
        total_score += momentum_score

        # 5. Final Aggregation
        total_score += partial_month_penalty
        
        if self.is_critical and total_score > 0:
            total_score += self.cfg.SCORE_CRITICAL_BONUS 
            
        total_score = min(100, total_score)
        
        # [Safe Zone Logic - Phase 3.1 Refined]
        # 조건부 Safe Zone: 점수가 아무리 높아도, 
        # (1) 건수가 적고 (2) Z-score가 낮으며 (3) **Nelson Rule 위반이 없을 때**만 0점 처리
        # Sparse 모드에서는 Z-score가 없으므로 건수 기준만 적용
        if not self.is_sparse:
            # 안전지대 조건: 건수 3건 미만 AND 시그마 0.8 미만
            is_in_safe_range = (self.current_value < 3) and (z_score_val < 0.8)
            # 패턴 위반 여부 (Bias, Trend 등)
            has_pattern_issue = len(triggered_rules) > 0
            
            # 범위는 안전하지만 패턴 이슈가 있다면 -> 점수 유지 (경고)
            # 범위도 안전하고 패턴 이슈도 없다면 -> 0점 (안전)
            if is_in_safe_range and not has_pattern_issue:
                # 단, 조기 과다(partial_reason)가 있다면 무시 못함
                if not partial_reason:
                    return {"score": 0, "status": "⚪", "reason": "정상범주"}

        # [Suppression] 1건 노이즈 필터링 (Phase 2.9 Logic - 수정: 일반 등급 첫 발생도 주의 경보로 제한)
        if self.current_value == 1:
            is_first_occurrence = (self.mean == 0)
            is_rare_breakout = (self.is_sparse and (len(self.history) > 0 and self.history.iloc[-1] == 0))
            
            if is_first_occurrence or is_rare_breakout:
                # 일반 등급 첫 발생은 위험 경보가 아닌 주의 경보 수준으로 제한
                if not self.is_critical:
                    total_score = min(total_score, self.cfg.THRESHOLD_YEL_GEN - 1)  # 49점으로 제한 (🟡)
                # 중대/위험 등급은 기존 로직 유지
            else:
                total_score = min(total_score, 30)
                partial_reason = "" 

        # 6. Status Determination & Text Consolidation (Phase 3.2)
        reason_parts = []
        if partial_reason: reason_parts.append(partial_reason)
        if triggered_rules: reason_parts.extend(triggered_rules)
        if not reason_parts: reason_parts.append(method_str)
        
        # Category-based Text Consolidation
        category_sudden = []      # ⚡돌발감지 (희소유형 발생 감지, 분포 이탈 감지)
        category_trend = []       # 📊추세이탈 (패턴 이탈 감지, 정상범위 이탈, 평균 대비 급증)
        category_momentum = []    # 📈급증감지 (지속적 상승, 연속 상승 모멘텀, 상승세 가속 감지)
        
        for part in reason_parts:
            if any(x in part for x in ["희소유형 돌발 발생", "분포 이탈"]):
                category_sudden.append(part)
            elif any(x in part for x in ["정상패턴 이탈", "정상범위 이탈", "평균 대비 급증"]):
                category_trend.append(part)
            elif any(x in part for x in ["지속적 상승", "연속 상승 모멘텀", "상승세 가속"]):
                category_momentum.append(part)
        
        # Build consolidated reason string
        reason_str = ""
        if category_sudden:
            details = ", ".join(category_sudden)
            reason_str = f"⚡돌발감지({details})"
        if category_trend:
            if reason_str: reason_str += " / "
            details = ", ".join(category_trend)
            reason_str += f"📊추세이탈({details})"
        if category_momentum:
            if reason_str: reason_str += " / "
            details = ", ".join(category_momentum)
            reason_str += f"📈급증감지({details})"
        
        # Fallback if nothing was categorized
        if not reason_str:
            reason_str = method_str if method_str else "정상범주"
        
        # Thresholds from Config
        thr_red = self.cfg.THRESHOLD_RED_CRIT if self.is_critical else self.cfg.THRESHOLD_RED_GEN
        thr_yel = self.cfg.THRESHOLD_YEL_CRIT if self.is_critical else self.cfg.THRESHOLD_YEL_GEN
        
        final_status = "⚪"
        if total_score >= thr_red:
            final_status = "🔴"
        elif total_score >= thr_yel:
            final_status = "🟡"

        if self.is_sparse and self.current_value == 2 and self.is_critical and self.mean >= 0.5:
            final_status = "🟡"

        if final_status == "⚪":
             return {"score": int(total_score), "status": "⚪", "reason": "정상범주"}
        else:
             return {"score": int(total_score), "status": final_status, "reason": reason_str}

# ============================================================================
# [NEW] Phase 2.5: Zero-Filling Data Preparation
# ============================================================================
def prepare_risk_data(
    df: pd.DataFrame,
    pivot_keys: List[str],
    target_date: Union[datetime, date, str],
    lookback_months: int = 24
) -> pd.DataFrame:
    """
    리스크 분석을 위한 데이터 전처리 (Zero-Filling & Pivot).
    
    동작:
        1. target_date 기준 과거 N개월 날짜 인덱스 생성
        2. pivot_keys를 기준으로 데이터 그룹화 및 집계
        3. 모든 그룹에 대해 N개월 기간의 빈 데이터를 0으로 채움 (Zero-filling)
        4. 리스크 엔진이 사용하기 편한 Wide Format (Columns=Date)으로 반환
    
    Args:
        df: 원본 데이터프레임 (접수일자 컬럼 필수)
        pivot_keys: 그룹핑할 컬럼 리스트 (예: ['등급기준', '대분류', '소분류'])
        target_date: 분석 기준일 (End Date)
        lookback_months: 과거 조회 기간 (기본 24개월)
        
    Returns:
        pd.DataFrame: Index=MultiIndex(pivot_keys), Columns=DatetimeIndex
    """
    if df.empty:
        return pd.DataFrame()

    # 1. Target Date Parsing
    if isinstance(target_date, str):
        target_ts = pd.to_datetime(target_date)
    elif isinstance(target_date, date) and not isinstance(target_date, datetime):
        target_ts = pd.to_datetime(target_date)
    else:
        target_ts = target_date

    # 2. Date Range Generation (Zero-filling 기준)
    start_ts = target_ts - relativedelta(months=lookback_months)
    # 월의 1일로 조정하여 정확한 매칭 유도
    start_ts = start_ts.replace(day=1)
    target_ts = target_ts.replace(day=1)
    
    full_date_idx = pd.date_range(start=start_ts, end=target_ts, freq='MS')
    
    # 3. Data Filtering (Range) & Month Column
    # 날짜 범위가 벗어나는 데이터를 미리 제거하지 않음 (pivot후 reindex로 처리)
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['접수일자']):
        df['접수일자'] = pd.to_datetime(df['접수일자'])
        
    df['__risk_month'] = df['접수일자'].dt.to_period('M').dt.to_timestamp()
    
    # 4. Pivot Table (Keys + Month -> Count)
    # pivot_table을 사용하여 존재하지 않는 조합은 제외하고, 존재하는 조합에 대해서만 카운트
    try:
        pivot = pd.pivot_table(
            df,
            index=pivot_keys,
            columns='__risk_month',
            values='상담번호', # Count 대상 (임의 컬럼)
            aggfunc='count',
            fill_value=0
        )
        
        # 5. Zero-Filling (Reindex Columns)
        # 생성된 피벗 테이블의 컬럼(날짜)을 강제로 full_date_idx로 맞춤
        # 없는 날짜는 0으로 채워짐
        filled_pivot = pivot.reindex(columns=full_date_idx, fill_value=0)
        
        return filled_pivot
        
    except Exception as e:
        print(f"[WARNING] prepare_risk_data 실패: {e}")
        return pd.DataFrame()


def calculate_lag_stats(df: pd.DataFrame) -> dict:
    """
    Calculate lag statistics from a DataFrame with 'Lag_Days' and 'Lag_Valid' columns.
    Returns dict with count, mean, median (p50), min, max, std.
    """
    if df is None or 'Lag_Days' not in df.columns or 'Lag_Valid' not in df.columns:
        return {'count': 0}
    valid_lag = df[df['Lag_Valid'] == True]['Lag_Days'].dropna()
    if valid_lag.empty:
        return {'count': 0}
    return {
        'count': int(valid_lag.count()),
        'mean': float(valid_lag.mean()),
        'p50': float(valid_lag.median()),
        'min': float(valid_lag.min()),
        'max': float(valid_lag.max()),
        'std': float(valid_lag.std()),
    }

# UI Wrapper
def calculate_advanced_risk_score(history_series: pd.Series, target_month_str: str, grade: str = None) -> Tuple[str, int, str]:
    try:
        if not isinstance(history_series.index, pd.DatetimeIndex):
            history_series.index = pd.to_datetime(history_series.index)
        target_ts = pd.to_datetime(target_month_str)
        
        # [Phase 2.5 Fix] prepare_risk_data를 통해 이미 Zero-filling된 데이터가 들어오므로
        # 여기서는 단순히 범위 내 데이터인지 확인만 하면 됨.
        # 혹시 모를 안전장치로 reindex 한 번 더 수행 가능하나, 성능상 생략.
        
        if target_ts in history_series.index:
            relevant_data = history_series.loc[:target_ts]
        else:
            # target_ts가 인덱스에 없으면 (Zero-filling이 안 된 Raw 데이터가 들어온 경우)
            # 여기서는 fallback으로 0 반환
            return "🟢", 0, "데이터 범위 오류"
            
        engine = RiskScoringEngine(relevant_data, grade=grade, target_month_str=target_month_str)
        result = engine.calculate_score()
        return result['status'], result['score'], result['reason']
    except Exception as e:
        return "⚪", 0, f"Err({str(e)})"
    
# --- [Utility Function] Missing in Phase 3.1 ---

def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using IQR method.
    Returns a boolean DataFrame where True indicates an outlier.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be DataFrame")
    
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return pd.DataFrame(False, index=df.index, columns=df.columns)
    
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    is_outlier = (numeric_df < lower_bound) | (numeric_df > upper_bound)
    
    result = pd.DataFrame(False, index=df.index, columns=df.columns)
    result[is_outlier.columns] = is_outlier
    return result
