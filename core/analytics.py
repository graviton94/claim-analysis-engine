"""
core/analytics.py
=================
Advanced Analytics Module for Food Safety Risk Scoring (v3.0).
Implementation Level: Phase 2.9 (Statistical Refinement)
 - CUSUM (Cumulative Sum) Algorithm Added
 - Weighted Scoring Matrix (Grade x Trend)
 - Volatility Adjustment (CV-based Dampening)
 - Zero-Filling Data Preparation (Preserved)
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom, linregress
from typing import Dict, Tuple, List, Union, Optional
import statsmodels.api as sm
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass
import calendar

# --- 1. Configuration Management (Refined) ---
@dataclass
class RiskConfig:
    # Data Requirements
    MIN_DATA_POINTS: int = 3
    MIN_SAMPLE_FOR_NB: int = 6  
    
    # Statistical Parameters
    EWMA_LAMBDA: float = 0.2
    
    # CUSUM Parameters (New)
    CUSUM_SLACK_STD: float = 0.5   # k (허용 오차: 0.5 시그마)
    CUSUM_DECISION_STD: float = 4.0 # h (판정 한계: 4.0 시그마)
    
    # Nelson Rules Windows
    NELSON_WINDOW_BIAS: int = 9
    NELSON_WINDOW_TREND: int = 6
    STL_MIN_PERIODS: int = 24
    
    # Base Scores
    SCORE_ACCIDENT: int = 100      # 사고 등급 즉시 만점
    
    # Scoring Component (Adders)
    SCORE_NELSON_DEV: int = 25     # 3시그마 이탈
    SCORE_NELSON_BIAS: int = 15    # 평균 이동
    SCORE_NELSON_TREND: int = 15   # 추세 지속
    SCORE_CUSUM_SHIFT: int = 20    # [New] CUSUM 이탈
    SCORE_MOMENTUM: int = 15       # 단기 급등
    
    SCORE_PARTIAL_ABS: int = 20    # 월중 절대치 초과
    SCORE_PARTIAL_VEL: int = 30    # 월중 속도 위반
    
    # Weight Matrix (Multipliers)
    # Key: Grade
    WEIGHT_GRADE = {
        '위험': 1.5,
        '중대': 1.2,
        '일반': 1.0,
        '미분류': 1.0
    }
    
    # Thresholds (Final)
    THRESHOLD_RED: int = 80
    THRESHOLD_YELLOW: int = 50
    
    # Safety Guards
    MIN_PROGRESS_FOR_VELOCITY: float = 0.2 

# Critical Grades Definition
CRITICAL_GRADES_SET = {'중대', '위험', '사고'}

class RiskScoringEngine:
    def __init__(self, data_series: pd.Series, grade: str = None, target_month_str: str = None):
        self.series = data_series.sort_index()
        self.grade = grade if grade else '미분류'
        self.is_critical = (self.grade in CRITICAL_GRADES_SET)
        self.cfg = RiskConfig()
        
        # 현재 월(Partial Month) 여부 확인
        self.is_partial_month = False
        self.progress_ratio = 1.0
        
        if target_month_str:
            try:
                today = datetime.now()
                target_date = datetime.strptime(target_month_str, "%Y-%m")
                if (target_date.year == today.year) and (target_date.month == today.month):
                    self.is_partial_month = True
                    day_of_month = max(1, today.day)
                    days_in_month = (target_date.replace(month=target_date.month % 12 + 1, day=1) - pd.Timedelta(days=1)).day
                    self.progress_ratio = day_of_month / days_in_month
            except:
                pass 

        if len(self.series) > 0:
            self.current_value = self.series.iloc[-1]
            self.history = self.series.iloc[:-1]
        else:
            self.current_value = 0
            self.history = pd.Series(dtype=float)
            
        self.n_obs = len(self.history)
        self.mean = self.history.mean() if self.n_obs > 0 else 0.0
        self.std = self.history.std() if self.n_obs > 1 else 0.0
        self.var = self.history.var() if self.n_obs > 1 else 0.0
        
        # 변동계수 (CV) 계산 for Volatility Adjustment
        self.cv = (self.std / self.mean) if self.mean > 0 else 0.0
        
        # [Regime] 희소성 판단
        zero_ratio = (self.history == 0).sum() / self.n_obs if self.n_obs > 0 else 0
        self.is_sparse = (self.mean < 1.0) or (zero_ratio > 0.5)

    def _calculate_cusum_score(self) -> float:
        """ [New] CUSUM (Cumulative Sum) Logic for Mean Shift Detection """
        if self.n_obs < 5 or self.std == 0: return 0.0
        
        # Standardize history
        z_hist = (self.history - self.mean) / self.std
        
        # Tabular CUSUM (Upper side only for risk)
        k = self.cfg.CUSUM_SLACK_STD
        h = self.cfg.CUSUM_DECISION_STD
        
        s_hi = 0
        triggered = False
        
        for z in z_hist:
            s_hi = max(0, s_hi + z - k)
            if s_hi > h:
                triggered = True
                # 한번 트리거 되면 루프 종료하지 않고 계속 상태 유지할 수도 있으나,
                # 여기서는 '과거에 이미 이탈했음'을 감지
        
        # 현재 값까지 포함하여 재확인
        z_curr = (self.current_value - self.mean) / self.std
        s_curr = max(0, s_hi + z_curr - k)
        
        if s_curr > h:
            return self.cfg.SCORE_CUSUM_SHIFT
        
        return 0.0

    def _calculate_sparse_score(self) -> Tuple[float, str]:
        """ 희소 데이터 스코어링 (Poisson/Negative Binomial) """
        if self.mean == 0:
            # 완전 신규 발생
            raw_score = 80.0 if self.current_value > 0 else 0.0
            method = "희소유형 돌발 발생"
        else:
            use_nbinom = False
            if self.n_obs >= self.cfg.MIN_SAMPLE_FOR_NB:
                if self.var > (1.2 * self.mean): use_nbinom = True
            
            if use_nbinom:
                p_est = self.mean / self.var
                r_est = (self.mean * p_est) / (1 - p_est)
                p_val = 1 - nbinom.cdf(self.current_value - 1, n=r_est, p=p_est)
                method = "분포 이탈(과대산포)"
            else:
                p_val = 1 - poisson.cdf(self.current_value - 1, mu=self.mean)
                method = "분포 이탈"

            if p_val < 1e-5: raw_score = 100.0
            else:
                raw_score = -np.log10(p_val) * 25
                raw_score = min(100.0, max(0.0, raw_score))

        return raw_score, method

    def _get_z_score_with_stl(self) -> Tuple[float, str]:
        """ Z-Score (STL or Standard) """
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
                return z, "계절성 이탈"
            except:
                pass
        
        z = (self.current_value - self.mean) / (self.std + 1e-6)
        return z, "평균 이탈"

    def calculate_score(self) -> Dict:
        # 0. 발생 없음
        if self.current_value == 0:
            return {"score": 0, "status": "🟢", "reason": "발생 없음"}

        # 1. Partial Month Penalty
        partial_penalty = 0
        partial_reason = ""
        
        if self.is_partial_month and self.current_value > 1:
            if self.current_value >= self.mean and self.mean > 0 and self.progress_ratio < 0.7:
                partial_penalty = self.cfg.SCORE_PARTIAL_ABS
                partial_reason = "조기 과다"
                
                # [Phase 2.9 Refinement] 희소 유형에 대한 조기 과다 가중치 완화
                if self.is_sparse:
                     partial_penalty = partial_penalty * 0.5 # 50% 감점
                     
            elif self.progress_ratio >= self.cfg.MIN_PROGRESS_FOR_VELOCITY:
                expected = self.mean * self.progress_ratio
                if expected > 0.5 and self.current_value > (expected * 3.0):
                    partial_penalty = self.cfg.SCORE_PARTIAL_VEL
                    partial_reason = "속도 위반"

        # 2. Main Scoring
        base_score = 0.0
        adders = 0.0
        reasons = []
        if partial_reason: reasons.append(partial_reason)

        # 3. Data Scarcity Check
        if self.n_obs < self.cfg.MIN_DATA_POINTS:
            # 데이터 부족 시 Rule-based
            if self.is_critical and self.current_value >= 2:
                return {"score": 100, "status": "🔴", "reason": f"초기급증({partial_reason})"}
            elif self.current_value >= 3:
                return {"score": 60, "status": "🟡", "reason": f"초기주의({partial_reason})"}
            else:
                return {"score": 0, "status": "⚪", "reason": "데이터 부족"}

        # 4. Calculation Logic (Sparse vs Dense)
        if self.is_sparse:
            # Sparse Logic
            prob_score, method = self._calculate_sparse_score()
            base_score = prob_score
            if prob_score > 50: reasons.append(method)
            
            # Trend Check in Sparse
            prev_val = self.history.iloc[-1] if len(self.history) > 0 else 0
            if prev_val > 0 and (self.current_value / prev_val) >= 2.0 and self.current_value >= 3:
                adders += self.cfg.SCORE_MOMENTUM
                reasons.append("연속 상승")
            
            # [Phase 2.9 Refinement] 일반 등급 희소 발생 허들 조정
            # 일반 등급이면서 희소 발생인 경우 점수 억제
            if not self.is_critical:
                 # 5건 미만이면 Danger(80점) 도달 방지
                 if self.current_value < 5:
                     base_score = min(base_score, 79)
                 
                 # 3건 미만이면 Caution(50점) 도달 방지 
                 if self.current_value < 3:
                     base_score = min(base_score, 49)

        else:
            # Dense Logic (SPC)
            z_score, z_method = self._get_z_score_with_stl()
            
            # Volatility Adjustment (CV 기반 임계값 완화)
            # CV가 높을수록(불안정) Z-score 임계값을 높임
            limit_z = 3.0
            if self.cv > 0.5: limit_z = 4.0 # 변동성 크면 4시그마 기준
            elif self.cv < 0.1: limit_z = 2.5 # 매우 안정적이면 2.5시그마 기준
            
            # Base Score from Z-score (Sigmoid-like mapping)
            if z_score > 1.0:
                base_score = min(50, (z_score - 1.0) * 20)
            
            # Nelson Rule 1 (Outlier)
            if z_score > limit_z:
                adders += self.cfg.SCORE_NELSON_DEV
                reasons.append(z_method)
            
            # Nelson Rule 2, 3 (Bias, Trend)
            full = self.series
            if len(full) >= self.cfg.NELSON_WINDOW_BIAS:
                if (full.iloc[-self.cfg.NELSON_WINDOW_BIAS:] > self.mean).all():
                    adders += self.cfg.SCORE_NELSON_BIAS
                    reasons.append("지속적 편차")
            
            if len(full) >= self.cfg.NELSON_WINDOW_TREND:
                diffs = full.iloc[-self.cfg.NELSON_WINDOW_TREND:].diff().dropna()
                if (diffs > 0).all():
                    adders += self.cfg.SCORE_NELSON_TREND
                    reasons.append("지속적 상승")
            
            # [New] CUSUM Check
            cusum_score = self._calculate_cusum_score()
            if cusum_score > 0:
                adders += cusum_score
                reasons.append("누적 합계 이탈")

        # 5. Final Aggregation with Weights
        total_raw = base_score + adders + partial_penalty
        
        # Apply Grade Multiplier
        weight = self.cfg.WEIGHT_GRADE.get(self.grade, 1.0)
        final_score = total_raw * weight
        
        # Cap at 100
        final_score = min(100, final_score)
        
        # 6. Suppression (Noise Filter) - [Phase 2.9 Modified]
        # 일반 등급 1건 노이즈 필터링은 위쪽 Sparse 로직에 흡수되었으나,
        # Dense 모드에서도 1건 급증(평균이 매우 낮은 경우)을 방어하기 위해 유지
        if self.current_value == 1 and not self.is_critical:
            final_score = min(final_score, self.cfg.THRESHOLD_RED - 1)
            
        # 7. Determine Status
        status = "⚪"
        if final_score >= self.cfg.THRESHOLD_RED: status = "🔴"
        elif final_score >= self.cfg.THRESHOLD_YELLOW: status = "🟡"
        
        # Reason Formatting
        reason_str = " / ".join(reasons) if reasons else "정상범주"
        if status == "⚪": reason_str = "정상범주"
        
        return {
            "score": int(final_score),
            "status": status,
            "reason": reason_str
        }

# --- Utility Functions ---

def prepare_risk_data(
    df: pd.DataFrame,
    pivot_keys: List[str],
    target_date: Union[datetime, date, str],
    lookback_months: int = 24
) -> pd.DataFrame:
    """ Zero-Filling Data Preparation (Maintained) """
    if df.empty: return pd.DataFrame()

    if isinstance(target_date, str):
        target_ts = pd.to_datetime(target_date)
    elif isinstance(target_date, date) and not isinstance(target_date, datetime):
        target_ts = pd.to_datetime(target_date)
    else:
        target_ts = target_date

    start_ts = target_ts - relativedelta(months=lookback_months)
    start_ts = start_ts.replace(day=1)
    target_ts = target_ts.replace(day=1)
    
    full_date_idx = pd.date_range(start=start_ts, end=target_ts, freq='MS')
    
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['접수일자']):
        df['접수일자'] = pd.to_datetime(df['접수일자'])
        
    df['__risk_month'] = df['접수일자'].dt.to_period('M').dt.to_timestamp()
    
    try:
        pivot = pd.pivot_table(
            df,
            index=pivot_keys,
            columns='__risk_month',
            values='상담번호',
            aggfunc='count',
            fill_value=0
        )
        filled_pivot = pivot.reindex(columns=full_date_idx, fill_value=0)
        return filled_pivot
    except Exception as e:
        print(f"[WARNING] prepare_risk_data 실패: {e}")
        return pd.DataFrame()

def calculate_lag_stats(df: pd.DataFrame) -> dict:
    """ Lag Statistics (Maintained) """
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

def calculate_advanced_risk_score(history_series: pd.Series, target_month_str: str, grade: str = None) -> Tuple[str, int, str]:
    """ Wrapper Function """
    try:
        if not isinstance(history_series.index, pd.DatetimeIndex):
            history_series.index = pd.to_datetime(history_series.index)
        target_ts = pd.to_datetime(target_month_str)
        
        if target_ts in history_series.index:
            relevant_data = history_series.loc[:target_ts]
        else:
            return "🟢", 0, "데이터 범위 오류"
            
        engine = RiskScoringEngine(relevant_data, grade=grade, target_month_str=target_month_str)
        result = engine.calculate_score()
        return result['status'], result['score'], result['reason']
    except Exception as e:
        return "⚪", 0, f"Err({str(e)})"

def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """ Outlier Detection (Maintained) """
    if not isinstance(df, pd.DataFrame): raise TypeError("Input must be DataFrame")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty: return pd.DataFrame(False, index=df.index, columns=df.columns)
    
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    is_outlier = (numeric_df < lower_bound) | (numeric_df > upper_bound)
    result = pd.DataFrame(False, index=df.index, columns=df.columns)
    result[is_outlier.columns] = is_outlier
    return result


# ============================================================================
# Phase 3.0: ForecastRiskAnalyzer (예측 시뮬레이션용 진단 엔진)
# ============================================================================

class ForecastRiskAnalyzer:
    """
    [Phase 3.0 Hybrid] 예측 시뮬레이션용 정밀 진단 엔진
    
    특징:
    1. Statistical Rigor: 기존 엔진의 회귀분석(Slope), 가속도(Acceleration), 변동성(CV) 로직 계승
    2. Structural Analysis: 신규 요구사항인 데이터 6분할 및 레벨 변동(Level Shift) 분석 적용
    3. Reality Check (Upgraded): 
       - Time-Aware: 월 중 진행률(Progress Ratio)을 반영하여 당월 예측값을 보정 후 비교
       - Noise Filter: 미미한 절대 수량 차이(5 미만)에 의한 과도한 알람 방지
    """
    
    def __init__(self, 
                 historical_series: pd.Series, 
                 current_actual: float,
                 current_forecast: float,
                 forecast_series: pd.Series,
                 grade: str = '미분류',
                 reference_date: Optional[datetime] = None):
        
        # 1. Data Segregation
        self.hist = historical_series.sort_index() if not historical_series.empty else pd.Series(dtype=float)
        self.curr_act = float(current_actual)
        self.curr_fcst = float(current_forecast)
        self.future = forecast_series.sort_index() if not forecast_series.empty else pd.Series(dtype=float)
        self.grade = grade
        self.ref_date = reference_date if reference_date else datetime.now()
        
        # 2. Statistics Baseline
        self.hist_mean = self.hist.mean() if not self.hist.empty else 0.0
        self.hist_std = self.hist.std() if len(self.hist) > 1 else 0.0
        self.hist_max = self.hist.max() if not self.hist.empty else 0.0
        
        self.future_mean = self.future.mean() if not self.future.empty else 0.0
        self.future_max = self.future.max() if not self.future.empty else 0.0
        
        # 3. Confidence Factor
        self.data_sufficiency = min(1.0, (len(self.hist) + len(self.future)) / 24.0)

    # --- [Core Math Methods] ---
    def _calculate_slope(self, series: pd.Series) -> float:
        """선형 회귀 기울기 계산"""
        if len(series) < 2: return 0.0
        try:
            y = series.values.astype(float)
            x = np.arange(len(y))
            slope, _, _, _, _ = linregress(x, y)
            return float(slope)
        except:
            return 0.0
    
    def _calculate_acceleration(self, series: pd.Series) -> float:
        """가속도 계산"""
        if len(series) < 4: return 0.0
        mid = len(series) // 2
        slope_first = self._calculate_slope(series.iloc[:mid])
        slope_second = self._calculate_slope(series.iloc[mid:])
        return slope_second - slope_first

    def _calculate_volatility(self, series: pd.Series) -> float:
        """변동성 (CV)"""
        if series.mean() == 0: return 0.0
        return series.std() / series.mean() if series.mean() > 0 else 0.0

    # --- [New Logic: Hybrid Analysis] ---

    def _analyze_trend_structure(self) -> Dict:
        """[Hybrid Logic] 추세(Trend)와 구조(Structure) 동시 분석"""
        # 1. Level Shift
        level_ratio = 0.0
        if self.hist_mean > 0:
            level_ratio = (self.future_mean - self.hist_mean) / self.hist_mean
            
        # 2. Trend Dynamics
        mean_scale = max(self.hist_mean, 1.0)
        future_slope = self._calculate_slope(self.future) / mean_scale
        acceleration = self._calculate_acceleration(self.future) / mean_scale
        
        score = 0
        pattern = "보합"
        
        if level_ratio > 0.5:
            pattern = "구조적 급등"
            score += 40
        elif level_ratio > 0.2:
            pattern = "레벨 상승"
            score += 25
            
        if future_slope > 0.1:
            if acceleration > 0.05:
                pattern += " (가속)"
                score += 15
            else:
                pattern += " (지속)"
                score += 10
        elif future_slope < -0.1:
            pattern = "하락 반전" if level_ratio > 0 else "하락세"
            
        return {"pattern": pattern, "score": score, "level_ratio": level_ratio}

    def _analyze_reality_gap(self) -> Dict:
        """
        [Upgraded Logic] 현실 괴리율 (Reality Gap)
        1. Time-Aware: 월 진행률(일자 기준)을 반영하여 '현재 시점의 기대 예측치' 산출
        2. Noise Filter: 절대 차이가 작으면(5 미만) 비율이 높아도 무시
        """
        # A. 진행률 계산 (Current Day / Days in Month)
        try:
            year, month = self.ref_date.year, self.ref_date.month
            _, last_day = calendar.monthrange(year, month)
            current_day = self.ref_date.day
            # 진행률 (최소 3% ~ 최대 100%)
            progress_ratio = max(0.03, min(1.0, current_day / last_day))
        except:
            progress_ratio = 1.0

        # B. 타겟 예측값 보정 (당월 총 예측값 * 진행률)
        target_fcst = self.curr_fcst * progress_ratio
        
        gap_score = 0
        status = "적정"
        gap_ratio = 0.0
        
        # C. Noise Filter (절대량 5건 미만 차이는 무시)
        abs_diff = self.curr_act - target_fcst
        if abs(abs_diff) < 5.0:
            return {
                "gap_ratio": 0.0, 
                "gap_score": 0, 
                "status": "적정 (미미함)", 
                "target_fcst": target_fcst
            }

        # D. 괴리율 계산 (Zero Handling)
        if target_fcst < 1.0: 
            # 예측은 0인데 실적이 유의미하게(5건 이상) 발생한 경우
            if self.curr_act >= 5.0:
                 gap_ratio = 5.0 # Cap ratio
                 status = "돌발 발생 (예측없음)"
                 gap_score = 30
        else:
            gap_ratio = (self.curr_act - target_fcst) / target_fcst

        # E. 리스크 판정 (실적이 예측 속도를 크게 상회할 때만 점수 부여)
        if gap_ratio > 0.5: # 50% 이상 초과 Pace
            gap_score = 30
            status = f"예측 괴리 심각(가속 {gap_ratio*100:.0f}%)"
        elif gap_ratio > 0.2: # 20% 이상 초과
            gap_score = 15
            status = "실적 상회"
            
        return {"gap_ratio": gap_ratio, "gap_score": gap_score, "status": status, "target_fcst": target_fcst}

    def _analyze_volatility_and_peak(self) -> Dict:
        """[Hybrid Logic] 변동성 및 전고점 돌파"""
        vol_score = 0
        signals = []
        
        # 1. 3-Sigma Breach
        if self.hist_std > 0:
            threshold = self.hist_mean + (3 * self.hist_std)
            if self.future_max > threshold:
                vol_score += 20
                signals.append("3σ 임계초과")
                
        # 2. Historical Peak Breach
        if self.hist_max > 0 and self.future_max > self.hist_max:
            breach_ratio = (self.future_max - self.hist_max) / self.hist_max
            if breach_ratio > 0.2:
                vol_score += 15
                signals.append(f"전고점 갱신(+{breach_ratio*100:.0f}%)")
                
        # 3. Volatility Expansion
        hist_cv = self._calculate_volatility(self.hist)
        future_cv = self._calculate_volatility(self.future)
        if future_cv > hist_cv * 1.5:
            vol_score += 10
            signals.append("변동성 확대")
            
        return {"vol_score": vol_score, "signals": signals}

    def analyze(self) -> Dict:
        """종합 진단 수행"""
        # 1. Component Analysis
        trend_res = self._analyze_trend_structure()
        gap_res = self._analyze_reality_gap()
        vol_res = self._analyze_volatility_and_peak()
        
        # 2. Base Score Calculation
        total_score = trend_res['score'] + gap_res['gap_score'] + vol_res['vol_score']
        
        # 3. Grade Weighting
        weight = 1.25 if self.grade in ['위험', '중대'] else 1.0
        final_score = min(100, int(total_score * weight))
        
        # 4. Status Determination
        if final_score >= 80:
            icon = "🔴"
            desc = "위험"
        elif final_score >= 50:
            icon = "🟡"
            desc = "주의"
        else:
            icon = "🟢"
            desc = "안정"
            
        # 5. Insight Generation
        insights = []
        if gap_res['gap_score'] > 0:
            insights.append(f"실적 페이스 {gap_res['gap_ratio']*100:.0f}% 초과")
        
        if trend_res['level_ratio'] > 0.2:
            insights.append(f"레벨 {trend_res['level_ratio']*100:.0f}% 상승 예상")
        elif trend_res['pattern'] != "보합":
            insights.append(f"{trend_res['pattern']}")
            
        if vol_res['signals']:
            insights.append(f"{vol_res['signals'][0]}")
            
        insight_text = " | ".join(insights) if insights else "특이사항 없음"

        return {
            "status": icon,
            "score": max(0, final_score), # Ensure non-negative
            "trend": trend_res['pattern'],
            "insight": insight_text,
            "details": {
                "level_shift": trend_res['level_ratio'],
                "reality_gap": gap_res['gap_ratio'],
                "volatility_signals": vol_res['signals']
            },
            "components": {
                "trend_score": trend_res['score'],
                "gap_score": gap_res['gap_score'],
                "vol_score": vol_res['vol_score']
            }
        }