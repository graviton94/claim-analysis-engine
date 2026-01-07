"""
core/analytics.py
=================
Advanced Analytics Module for Food Safety Risk Scoring.
Implementation Level: Phase 2.6 (Sensitivity Tuning + Partial Month Logic)
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
from typing import Dict, Tuple, List
import statsmodels.api as sm
from datetime import datetime

# --- Configuration Constants ---
MIN_DATA_POINTS = 3          
EWMA_LAMBDA = 0.2            
NELSON_WINDOW_BIAS = 9       
NELSON_WINDOW_TREND = 6      
STL_MIN_PERIODS = 24         

# Critical Grades Definition (for Sensitivity)
CRITICAL_GRADES_SET = {'중대', '위험', '사고'}

class RiskScoringEngine:
    def __init__(self, data_series: pd.Series, grade: str = None, target_month_str: str = None):
        self.series = data_series.sort_index()
        self.grade = grade
        self.is_critical = (grade in CRITICAL_GRADES_SET) if grade else False
        
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
                    # 진행률 계산 (예: 30일 중 15일 지남 -> 0.5)
                    # (안전장치: 최소 1일은 지난 것으로 간주)
                    day_of_month = max(1, today.day)
                    days_in_month = (target_date.replace(month=target_date.month % 12 + 1, day=1) - pd.Timedelta(days=1)).day
                    self.progress_ratio = day_of_month / days_in_month
            except:
                pass # 날짜 파싱 에러 시 기본값(Full Month) 유지

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
        """ [Track A] 희소 데이터 """
        if self.mean == 0:
            raw_score = 100.0 if self.current_value > 0 else 0.0
            method = "첫 발생"
        else:
            is_over_dispersed = self.var > (1.2 * self.mean)
            if is_over_dispersed and self.n_obs > 5:
                p_est = self.mean / self.var
                r_est = (self.mean * p_est) / (1 - p_est)
                p_val = 1 - nbinom.cdf(self.current_value - 1, n=r_est, p=p_est)
                method = "NB"
            else:
                p_val = 1 - poisson.cdf(self.current_value - 1, mu=self.mean)
                method = "Poisson"

            if p_val < 1e-5: raw_score = 100.0
            else:
                raw_score = -np.log10(p_val) * 25
                raw_score = min(100.0, max(0.0, raw_score))

        # [Correction] 절대 건수 보정 (1건은 약하게)
        if self.current_value == 1:
            decay = 0.5
            # 단, 중대 등급이거나 부분월 조기 경보인 경우 감쇠 완화
            if self.is_critical or (self.is_partial_month and self.progress_ratio < 0.2):
                decay = 0.8 
        elif self.current_value == 2:
            decay = 0.8
            if self.is_critical: decay = 1.0
        else:
            decay = 1.0
            
        return raw_score * decay, method

    def _get_z_score_with_stl(self) -> Tuple[float, str]:
        """ [Track B] Z-Score (STL or Standard) """
        if self.n_obs >= STL_MIN_PERIODS and self.std > 0:
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
                return z, "이상패턴"
            except:
                pass
        
        z = (self.current_value - self.mean) / (self.std + 1e-6)
        return z, "편차이탈"

    def _apply_nelson_rules(self, z_score: float) -> Tuple[float, List[str]]:
        triggered = []
        score_add = 0.0
        full = self.series
        
        # [Tuning] 중대 등급이면 임계값 강화
        limit_z = 2.5 if self.is_critical else 3.0
        warn_z = 1.5 if self.is_critical else 2.0
        
        if abs(z_score) > limit_z:
            triggered.append(f"관리이탈(>{limit_z}σ)")
            score_add += 30
        elif abs(z_score) > warn_z:
            score_add += 15
            
        if len(full) >= NELSON_WINDOW_BIAS:
            last_n = full.iloc[-NELSON_WINDOW_BIAS:]
            if (last_n > self.mean).all():
                triggered.append(f"평균이동({NELSON_WINDOW_BIAS}M)")
                score_add += 20
                
        if len(full) >= NELSON_WINDOW_TREND:
            last_n = full.iloc[-NELSON_WINDOW_TREND:]
            diffs = last_n.diff().dropna()
            if (diffs > 0).all():
                triggered.append(f"지속상승({NELSON_WINDOW_TREND}M)")
                score_add += 20
                
        return score_add, triggered

    def _calculate_velocity_score(self) -> float:
        """ [New] 급격한 기울기 변화 감지 (일반 등급 미탐 방지) """
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
            return {"score": 0, "status": "", "reason": "발생 없음"}

        # 1. 월중 조기 경보 (Partial Month Logic)
        partial_month_penalty = 0
        partial_reason = ""
        
        if self.is_partial_month:
            # A. 절대 속도 위반 (이미 월평균 초과)
            if self.current_value >= self.mean and self.mean > 0:
                partial_month_penalty = 50 # 강력한 페널티
                partial_reason = f"조기과다({int(self.progress_ratio*100)}%감지)"
            # B. 상대 속도 위반 (평소보다 3배 이상 빠른 페이스)
            else:
                expected_current = self.mean * self.progress_ratio
                if expected_current > 0 and self.current_value > (expected_current * 3.0):
                    partial_month_penalty = 30
                    partial_reason = f"속도급증({int(self.progress_ratio*100)}%감지)"
                elif expected_current > 0 and self.current_value > (expected_current * 2.0):
                    # 중대 등급이면 2배 빨라도 경고
                    if self.is_critical:
                        partial_month_penalty = 20
                        partial_reason = "속도주의(중대)"

        # [Logic Check 1] Safe Zone (안전지대) - 중대 등급 오탐 방지
        # 현재 값이 '평균 + 0.8표준편차' 이내라면 무조건 정상
        # (단, 3건 이상 급증한 경우는 예외)
        safe_threshold = self.mean + (0.8 * self.std)
        if self.current_value <= safe_threshold and self.current_value < 3:
             return {"score": 0, "status": "⚪", "reason": "정상범주"}

        # 2. 데이터 부족 (초기)
        if self.n_obs < MIN_DATA_POINTS:
            # 중대 등급이면 1건이어도 초기엔 민감하게
            threshold_danger = 2 if self.is_critical else 3
            threshold_warn = 1 if self.is_critical else 2
            
            if self.current_value >= threshold_danger:
                return {"score": 100, "status": "🔴", "reason": f"초기급증({partial_reason})"}
            elif self.current_value >= threshold_warn:
                 return {"score": 50, "status": "🟡", "reason": f"초기주의({partial_reason})"}
            else:
                return {"score": 0, "status": "⚪", "reason": "데이터 부족"}

        # 3. Main Scoring (Track A/B)
        if self.is_sparse:
            prob_score, method = self._calculate_sparse_score()
            
            # Trend Check
            trend_score = 0
            prev_val = self.history.iloc[-1] if len(self.history) > 0 else 0
            if prev_val > 0 and (self.current_value / prev_val) >= 3.0 and self.current_value >= 3:
                trend_score = 20
            
            total_score = prob_score + trend_score
        else: # Dense
            z_score, z_method = self._get_z_score_with_stl()
            
            # [Sensitivity] 중대 등급이면 0.5 시그마부터 점수 부여
            start_sigma = 0.5 if self.is_critical else 1.0
            base_score = min(50, max(0, (z_score - start_sigma) * (50 / 2.0)))
            
            nelson_score, rules = self._apply_nelson_rules(z_score)
            
            # EWMA
            ewma = self.series.ewm(alpha=EWMA_LAMBDA, adjust=False).mean()
            z_ewma = (ewma.iloc[-1] - self.mean) / (self.std * np.sqrt(EWMA_LAMBDA/(2-EWMA_LAMBDA)) + 1e-6)
            ewma_score = 15.0 if abs(z_ewma) > 3.0 else 0.0
            
            # [New] Velocity Score 추가
            velocity_score = self._calculate_velocity_score()
            
            total_score = base_score + nelson_score + ewma_score + velocity_score
            method = z_method

        # 4. 최종 점수 합산 (Partial Penalty & Sensitivity)
        total_score += partial_month_penalty
        
        # [Sensitivity] 중대 등급 기본 가산점
        if self.is_critical and total_score > 0:
            total_score += 10 # 일단 발생하면 10점 깔고 시작
            
        total_score = min(100, total_score)
        
        # 5. 상태 결정
        reason_str = partial_reason if partial_reason else method
        if self.is_sparse and "NB" in method: reason_str = f"희소({method})"
        
        threshold_red = 75 if self.is_critical else 80
        threshold_yellow = 40 if self.is_critical else 50
        
        final_status = "⚪" # Default
        if total_score >= threshold_red:
            final_status = "🔴"
        elif total_score >= threshold_yellow:
            final_status = "🟡"

        # [Tuning] 중대 등급 2건일 때, 평균이 0.5 이상이면(덜 희귀하면) 🟡로 완화
        if self.is_sparse and self.current_value == 2 and self.is_critical and self.mean >= 0.5:
            final_status = "🟡"

        if final_status == "⚪":
             return {"score": int(total_score), "status": "⚪", "reason": "정상범주"}
        else:
             return {"score": int(total_score), "status": final_status, "reason": reason_str}


# UI Wrapper
def calculate_advanced_risk_score(history_series: pd.Series, target_month_str: str, grade: str = None) -> Tuple[str, int, str]:
    try:
        if not isinstance(history_series.index, pd.DatetimeIndex):
            history_series.index = pd.to_datetime(history_series.index)
        target_ts = pd.to_datetime(target_month_str)
        
        if target_ts in history_series.index:
            relevant_data = history_series.loc[:target_ts]
        else:
            return "🟢", 0, "당월0건"
            
        # 등급과 타겟월 정보를 엔진에 주입
        engine = RiskScoringEngine(relevant_data, grade=grade, target_month_str=target_month_str)
        result = engine.calculate_score()
        return result['status'], result['score'], result['reason']
    except Exception as e:
        return "⚪", 0, f"Err"

# Legacy Functions (유지)
def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame): raise TypeError("Input must be DataFrame")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty: return pd.DataFrame(False, index=df.index, columns=df.columns)
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    is_outlier = (numeric_df < lower) | (numeric_df > upper)
    result = pd.DataFrame(False, index=df.index, columns=df.columns)
    result[is_outlier.columns] = is_outlier
    return result

def calculate_lag_stats(df: pd.DataFrame, lag_col: str = 'Lag_Days') -> Dict:
    stats_keys = ['mean', 'std', 'min', 'max', 'p25', 'p50', 'p75', 'count']
    empty_stats = {key: 0 for key in stats_keys}
    if lag_col not in df.columns or 'Lag_Valid' not in df.columns: return empty_stats
    valid_lags = df.loc[df['Lag_Valid'] == True, lag_col].dropna()
    if valid_lags.empty: return empty_stats
    return {
        'mean': round(valid_lags.mean(), 1),
        'std': round(valid_lags.std(), 1),
        'min': int(valid_lags.min()),
        'max': int(valid_lags.max()),
        'p25': int(valid_lags.quantile(0.25)),
        'p50': int(valid_lags.quantile(0.50)),
        'p75': int(valid_lags.quantile(0.75)),
        'count': int(len(valid_lags))
    }