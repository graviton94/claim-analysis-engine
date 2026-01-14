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
    예측 시뮬레이션용 진단 엔진 (Phase 3.0)
    과거-현재-미래를 잇는 시계열 관점에서 리스크를 진단합니다.
    
    분석 기준:
    - 과거(Historical): 과거 추세
    - 현재(Current): 최근 월의 수준
    - 미래(Forecast): 예측 기간의 추세 및 수준
    """
    
    def __init__(self, 
                 historical_series: pd.Series,  # 과거 데이터 (최근 12개월 권장)
                 current_value: float,          # 당월 실적 (확정치 또는 예상치)
                 forecast_series: pd.Series,    # 예측 데이터 (향후 N개월)
                 grade: str = '미분류'):
        
        self.hist = historical_series.sort_index() if not historical_series.empty else pd.Series(dtype=float)
        self.curr = current_value
        self.fcst = forecast_series.sort_index() if not forecast_series.empty else pd.Series(dtype=float)
        self.grade = grade
        
        # 통계치 미리 계산 (과거만 기반)
        self.hist_mean = self.hist.mean() if not self.hist.empty else 0
        self.hist_std = self.hist.std() if len(self.hist) > 1 else 0
        self.hist_max = self.hist.max() if not self.hist.empty else 0
        self.hist_min = self.hist.min() if not self.hist.empty else 0
        self.hist_median = self.hist.median() if not self.hist.empty else 0
        
        # 미래 통계
        self.fcst_mean = self.fcst.mean() if not self.fcst.empty else 0
        self.fcst_std = self.fcst.std() if len(self.fcst) > 1 else 0
        self.fcst_max = self.fcst.max() if not self.fcst.empty else 0
        
        # 신뢰도 계산 (데이터 포인트 수)
        self.hist_data_pts = len(self.hist)
        self.fcst_data_pts = len(self.fcst)
        self.confidence_hist = min(1.0, self.hist_data_pts / 24.0)  # 24개월 기준
        self.confidence_fcst = min(1.0, self.fcst_data_pts / 12.0)  # 12개월 기준

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
        """가속도 계산 (기울기의 변화율)"""
        if len(series) < 4: return 0.0
        mid = len(series) // 2
        slope_first = self._calculate_slope(series.iloc[:mid])
        slope_second = self._calculate_slope(series.iloc[mid:])
        return slope_second - slope_first
    
    def _calculate_volatility(self, series: pd.Series) -> float:
        """변동성 (CV: Coefficient of Variation)"""
        if series.mean() == 0: return 0.0
        return series.std() / series.mean() if series.mean() > 0 else 0.0
    
    def _calculate_jump_rate(self, series: pd.Series) -> float:
        """점프율: 연속 두 값의 최대 변화율"""
        if len(series) < 2: return 0.0
        diffs = series.pct_change().dropna()
        return diffs.abs().max() if not diffs.empty else 0.0

    def _calculate_mean_delta_score(self) -> Tuple[float, str]:
        """과거 평균 대비 예측 평균 상승률을 점수화"""
        if self.hist_mean <= 0: return 0.0, ""
        ratio = self.fcst_mean / self.hist_mean if self.hist_mean > 0 else 0
        if ratio <= 1.0: return 0.0, ""
        if ratio >= 2.0: return 30.0, "예측 평균 2배↑"
        if ratio >= 1.5: return 20.0, "예측 평균 50%↑"
        if ratio >= 1.25: return 12.0, "예측 평균 25%↑"
        if ratio >= 1.1: return 5.0, "예측 평균 10%↑"
        return 0.0, ""
    
    def _analyze_trend_signals(self) -> Dict:
        """트렌드 신호 분석"""
        signals = {
            'past_slope': 0,
            'future_slope': 0,
            'acceleration': 0,
            'trend_text': '안정',
            'signal_count': 0  # 경고 신호 개수
        }
        
        # 최근/미래 기울기를 평균 규모로 정규화하여 민감도 개선
        mean_scale = max(self.hist_mean, 1.0)
        if len(self.hist) >= 4:
            signals['past_slope'] = self._calculate_slope(self.hist.tail(4)) / mean_scale
        
        if len(self.fcst) >= 3:
            signals['future_slope'] = self._calculate_slope(self.fcst.head(3)) / mean_scale
            signals['acceleration'] = self._calculate_acceleration(self.fcst) / mean_scale
        
        # 트렌드 분류 기준(정규화된 기울기 기반, 더 민감하게 조정)
        threshold_rising = 0.1
        threshold_falling = -0.1
        accel_threshold = 0.05
        
        if signals['future_slope'] > threshold_rising:
            if signals['past_slope'] > threshold_rising:
                signals['trend_text'] = '상승 가속' if signals['acceleration'] > accel_threshold else '상승 지속'
                signals['signal_count'] += 2
            else:
                signals['trend_text'] = '상승 반전'
                signals['signal_count'] += 1
        elif signals['future_slope'] < threshold_falling:
            if signals['past_slope'] > threshold_rising:
                signals['trend_text'] = '하락 반전'
                signals['signal_count'] += 1
            else:
                signals['trend_text'] = '하락 지속'
        else:
            signals['trend_text'] = '안정'
        
        return signals
    
    def _analyze_level_signals(self) -> Dict:
        """수준 신호 분석 (평균 기반)"""
        signals = {
            'level_text': '정상',
            'level_score': 0,
            'threshold_warning': self.hist_mean + (1.5 * self.hist_std) if self.hist_std > 0 else self.hist_mean * 1.5,
            'threshold_critical': self.hist_mean + (3.0 * self.hist_std) if self.hist_std > 0 else self.hist_mean * 3.0,
        }
        
        if self.fcst_mean > signals['threshold_critical']:
            signals['level_text'] = '초위험(3σ 초과)'
            signals['level_score'] = 85
        elif self.fcst_mean > signals['threshold_warning']:
            signals['level_text'] = '주의(1.5σ 초과)'
            signals['level_score'] = 50
        elif self.fcst_mean > self.curr * 1.2:  # 현재 대비 20% 상승
            signals['level_text'] = '소폭 상승'
            signals['level_score'] = 25
        else:
            signals['level_text'] = '정상'
            signals['level_score'] = 0
        
        return signals
    
    def _analyze_peak_signals(self) -> Dict:
        """최고점 신호 분석"""
        signals = {
            'is_new_record': False,
            'record_breach_ratio': 0.0,
            'peak_score': 0
        }
        
        if self.hist_max > 0:
            signals['is_new_record'] = self.fcst_max > self.hist_max
            signals['record_breach_ratio'] = (self.fcst_max - self.hist_max) / self.hist_max if self.hist_max > 0 else 0
            
            if signals['is_new_record']:
                if signals['record_breach_ratio'] > 0.5:  # 50% 이상 초과
                    signals['peak_score'] = 40
                else:
                    signals['peak_score'] = 25
        
        return signals
    
    def _analyze_volatility_signals(self) -> Dict:
        """변동성 신호 분석"""
        signals = {
            'hist_cv': self._calculate_volatility(self.hist) if not self.hist.empty else 0,
            'fcst_cv': self._calculate_volatility(self.fcst) if not self.fcst.empty else 0,
            'hist_jump': self._calculate_jump_rate(self.hist) if not self.hist.empty else 0,
            'fcst_jump': self._calculate_jump_rate(self.fcst) if not self.fcst.empty else 0,
            'volatility_score': 0
        }
        
        # 변동성 증가 감지
        if signals['fcst_cv'] > signals['hist_cv'] * 1.5:
            signals['volatility_score'] += 15  # 변동성 증가
        
        # 점프 위험 감지
        if signals['fcst_jump'] > 1.0:  # 100% 이상 변화
            signals['volatility_score'] += 20
        elif signals['fcst_jump'] > 0.5:  # 50% 이상 변화
            signals['volatility_score'] += 10
        
        return signals
    
    def _analyze_grade_impact(self, base_score: float) -> Tuple[float, str]:
        """등급별 가중치 적용"""
        weight_map = {
            '위험': 1.4,
            '중대': 1.25,
            '일반': 1.0,
            '미분류': 1.0
        }
        
        weight = weight_map.get(self.grade, 1.0)
        adjusted_score = base_score * weight
        
        grade_text = f"({self.grade} 등급)" if self.grade != '미분류' else ""
        
        return min(100, adjusted_score), grade_text

    def analyze(self) -> Dict:
        """
        종합 분석 수행
        
        반환:
        - status: 🔴/🟡/🟢 (위험도 아이콘)
        - score: 0~100 (예측 리스크 점수)
        - trend: 트렌드 분석 결과
        - signals: 개별 신호 상세 정보
        - insight: 종합 해석 텍스트
        - confidence: 분석 신뢰도 (0~1)
        """
        
        # 1. 모든 신호 수집
        trend_sig = self._analyze_trend_signals()
        level_sig = self._analyze_level_signals()
        peak_sig = self._analyze_peak_signals()
        vol_sig = self._analyze_volatility_signals()
        delta_score, delta_text = self._calculate_mean_delta_score()
        
        # 2. 기본 점수 계산
        base_score = 0
        
        # 수준 신호 (최대 85점)
        base_score += level_sig['level_score']
        base_score += delta_score
        
        # 트렌드 신호 (최대 30점)
        if trend_sig['trend_text'] == '상승 가속':
            base_score += 30
        elif trend_sig['trend_text'] == '상승 반전':
            base_score += 20
        elif trend_sig['trend_text'] == '상승 지속':
            base_score += 15
        
        # 최고점 신호 (최대 40점)
        base_score += peak_sig['peak_score']
        
        # 변동성 신호 (최대 20점)
        base_score += vol_sig['volatility_score']
        
        # 3. 등급 가중치 적용
        final_score, grade_text = self._analyze_grade_impact(base_score)
        
        # 4. 상태 아이콘 결정
        status_icon = "🟢"
        if final_score >= 80:
            status_icon = "🔴"
        elif final_score >= 50:
            status_icon = "🟡"
        
        # 5. 신뢰도 계산
        confidence = (self.confidence_hist + self.confidence_fcst) / 2
        
        # 6. 종합 Insight 생성
        insight_parts = []
        
        # 과거-현재 비교
        if self.curr > 0 and self.hist_mean > 0:
            curr_ratio = self.curr / self.hist_mean
            if curr_ratio > 1.3:
                insight_parts.append(f"현재 수준 +{(curr_ratio-1)*100:.0f}%")
            elif curr_ratio < 0.7:
                insight_parts.append(f"현재 수준 {(curr_ratio-1)*100:.0f}%")
        
        # 추세
        insight_parts.append(f"추세: {trend_sig['trend_text']}")
        
        # 미래 전망
        if self.fcst_mean > self.hist_mean:
            fcst_ratio = self.fcst_mean / self.hist_mean
            insight_parts.append(f"예측 평균 +{(fcst_ratio-1)*100:.0f}%")
        if delta_text:
            insight_parts.append(delta_text)
        
        # 최고점
        if peak_sig['is_new_record']:
            insight_parts.append(f"❗ 최고점 갱신 +{peak_sig['record_breach_ratio']*100:.0f}%")
        
        # 변동성
        if vol_sig['volatility_score'] > 15:
            insight_parts.append("⚠️ 변동성 증가")
        
        insight_str = " | ".join(insight_parts)
        
        # 7. 세부 신호 반환
        return {
            'status': status_icon,
            'score': int(final_score),
            'trend': trend_sig['trend_text'],
            'insight': insight_str,
            'grade_text': grade_text,
            'confidence': round(confidence, 2),
            'signals': {
                'trend': trend_sig,
                'level': level_sig,
                'peak': peak_sig,
                'volatility': vol_sig
            },
            'components': {
                'score_level': level_sig['level_score'],
                'score_mean_delta': delta_score,
                'score_trend': min(30, base_score - level_sig['level_score'] - peak_sig['peak_score'] - vol_sig['volatility_score']),
                'score_peak': peak_sig['peak_score'],
                'score_volatility': vol_sig['volatility_score']
            }
        }