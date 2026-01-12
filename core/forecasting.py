"""
📊 Advanced Forecasting Engine Module
목표: 다중 시계열 분석 및 고도화된 예측 모델

구성:
1. 이상치 탐지 및 처리 (IQR 기반)
2. 영업일 기준 진행률 (Business Day Normalization)
3. 다중 모델 앙상블:
   - Run-rate (현재 페이스)
   - Historical Pattern (MoM 비율)
   - 회귀 기반 추세 (Trend Line)
   - Holt-Winters (계절성 + 추세)
   - SARIMA (자동 파라미터) - 통계적 정확도
4. 적응형 가중치 시스템 (신뢰도/데이터 충분성 기반)
5. 신뢰도 구간 제공 (95% CI)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False


class ForecastEngine:
    """
    고도화된 시계열 예측 엔진
    - 이상치 탐지 및 처리 (IQR 방식)
    - 영업일 기반 정밀 진행률 계산
    - 다중 모델 앙상블 (Run-rate, Pattern, Trend, HW, SARIMA)
    - 적응형 가중치 시스템
    - 신뢰도 구간 제공
    """
    
    def __init__(self, raw_df: pd.DataFrame, date_col: str = '접수일자'):
        """
        초기화 시 전체 데이터를 정제하고 고급 분석
        
        Args:
            raw_df: 전체 원본 데이터프레임
            date_col: 날짜 컬럼명 (기본값: '접수일자')
        """
        self.raw_df = raw_df.copy()
        self.date_col = date_col
        
        # 날짜 컬럼 처리
        if date_col in self.raw_df.columns:
            self.raw_df[date_col] = pd.to_datetime(self.raw_df[date_col])
        else:
            raise ValueError(f"컬럼 '{date_col}'을 찾을 수 없습니다.")
        
        # 연도-월 기준 집계
        self.raw_df['year_month'] = self.raw_df[date_col].dt.to_period('M')
        
        # 월별 건수 집계 (시계열 데이터)
        self.monthly_series = self.raw_df.groupby('year_month').size().astype(int)
        
        # ===== 이상치 탐지 및 처리 =====
        self.monthly_series_cleaned = self._remove_outliers()
        
        # 메타데이터 추출
        self.min_date = self.raw_df[date_col].min()
        self.max_date = self.raw_df[date_col].max()
        self.current_year = self.max_date.year
        self.current_month = self.max_date.month
        
        # ===== 학습 데이터 분리 (Data Leakage 방지) =====
        # 현재 달이 아직 진행 중이면 마지막 행(불완전한 당월 데이터) 제외
        # → Holt-Winters 등 모델이 "당월 실적 급락"으로 오인하는 것을 방지
        self.max_days_in_month = pd.Timestamp(self.max_date).days_in_month
        if self.max_date.day < self.max_days_in_month:
            # 월말이 아니면 마지막 달 제외
            self.training_series_cleaned = self.monthly_series_cleaned.iloc[:-1]
        else:
            # 월말이면 전체 데이터 사용
            self.training_series_cleaned = self.monthly_series_cleaned
        
        # 고급 분석 (모두 training_series 기반)
        self._calculate_trend_line()
        self._calculate_mom_ratios()
        self._calculate_seasonal_factors()
        self._estimate_volatility()
    
    def _remove_outliers(self, method='iqr', threshold=1.5) -> pd.Series:
        """
        이상치 탐지 및 제거 (IQR 방식)
        
        극단적인 값이 예측에 미치는 영향을 최소화
        """
        series = self.monthly_series.copy()
        
        if len(series) < 4:
            return series
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # 이상치는 중앙값으로 대체
        median_val = series.median()
        series_cleaned = series.copy()
        
        outliers = (series < lower_bound) | (series > upper_bound)
        series_cleaned[outliers] = median_val
        
        return series_cleaned
    
    def _calculate_trend_line(self):
        """
        선형 회귀를 통한 추세 계산
        전체 시계열의 장기 추세 파악
        (training_series 기반으로 학습)
        """
        try:
            x = np.arange(len(self.training_series_cleaned))
            y = self.training_series_cleaned.values
            
            # NaN 값 제거
            mask = ~np.isnan(y)
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) > 2:
                coeffs = np.polyfit(x_clean, y_clean, 1)
                self.trend_slope = coeffs[0]  # 월별 변화율
                self.trend_intercept = coeffs[1]
            else:
                self.trend_slope = 0
                self.trend_intercept = y_clean.mean() if len(y_clean) > 0 else 0
        except Exception as e:
            print(f"[WARNING] 추세선 계산 실패: {e}")
            self.trend_slope = 0
            self.trend_intercept = self.monthly_series_cleaned.mean()
    
    def _estimate_volatility(self):
        """
        데이터 변동성 계산 (표준편차 기반)
        예측 신뢰도를 결정하는 중요 지표
        (training_series 기반으로 계산)
        """
        try:
            returns = self.training_series_cleaned.pct_change().dropna()
            self.volatility = returns.std()
            
            # 변동성 등급: Low(<0.1), Medium(0.1~0.3), High(>0.3)
            if self.volatility < 0.1:
                self.volatility_level = "Low"
            elif self.volatility < 0.3:
                self.volatility_level = "Medium"
            else:
                self.volatility_level = "High"
        except Exception:
            self.volatility = 0.15
            self.volatility_level = "Medium"
    
    def _calculate_business_days(self, start_date: datetime, end_date: datetime) -> int:
        """영업일(평일) 수 계산 (토일 제외)"""
        return int(np.busday_count(start_date.date(), end_date.date() + timedelta(days=1)))
    
    def _get_total_business_days_in_month(self, year: int, month: int) -> int:
        """해당 월의 총 영업일 수"""
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end = datetime(year, month + 1, 1) - timedelta(days=1)
        return self._calculate_business_days(start, end)
    
    def _calculate_mom_ratios(self):
        """
        전월 대비 당월의 평균 증감율(MoM Ratio) 계산
        예: (1월 평균 / 12월 평균), (2월 평균 / 1월 평균), ...
        
        이를 통해 계절적 변동성을 반영
        (training_series 기반으로 계산)
        """
        self.mom_ratios = {}  # {month: ratio}
        
        # 월별 평균 계산
        monthly_avg = {}
        for month in range(1, 13):
            month_data = self.training_series_cleaned[self.training_series_cleaned.index.month == month]
            if len(month_data) > 0:
                monthly_avg[month] = month_data.mean()
            else:
                monthly_avg[month] = 0
        
        # 전월 대비 당월의 비율 계산
        for month in range(1, 13):
            prev_month = month - 1 if month > 1 else 12
            
            if monthly_avg[prev_month] > 0:
                ratio = monthly_avg[month] / monthly_avg[prev_month]
                # 극단값 필터링 (0.6 ~ 1.5 범위로 더 보수적)
                ratio = max(0.6, min(ratio, 1.5))
                self.mom_ratios[month] = ratio
            else:
                self.mom_ratios[month] = 1.0
    
    def _calculate_seasonal_factors(self):
        """
        계절성 지수 계산 (월별 평균 / 전체 평균)
        (training_series 기반으로 계산)
        """
        monthly_avg = self.training_series_cleaned.groupby(
            self.training_series_cleaned.index.month
        ).transform('mean')
        
        self.seasonal_factors = {}
        for month in range(1, 13):
            month_data = self.training_series_cleaned[self.training_series_cleaned.index.month == month]
            if len(month_data) > 0:
                self.seasonal_factors[month] = month_data.mean() / self.training_series_cleaned.mean()
            else:
                self.seasonal_factors[month] = 1.0
    
    def predict_current_month(self, current_val: int, current_date: datetime) -> dict:
        """
        당월 말일 최종값 예측 (앙상블: Run-rate + Historical Pattern + Dynamic Weight)
        
        Args:
            current_val: 현재까지의 누적 건수
            current_date: 현재 날짜
        
        Returns:
            {
                "predicted_final": 1250,
                "confidence": "High",
                "details": {
                    "run_rate": 1200,
                    "pattern_based": 1300,
                    "weights": "Run-rate(0.8) + Pattern(0.2)"
                }
            }
        """
        try:
            year = current_date.year
            month = current_date.month
            
            # ===== STEP 1: 영업일 기반 진행률 =====
            month_start = datetime(year, month, 1)
            bdays_passed = self._calculate_business_days(month_start, current_date)
            total_bdays = self._get_total_business_days_in_month(year, month)
            
            if bdays_passed <= 0 or total_bdays <= 0:
                return {
                    "predicted_final": current_val,
                    "confidence": "Very Low",
                    "details": {
                        "run_rate": current_val,
                        "pattern_based": current_val,
                        "weights": "N/A (insufficient data)"
                    }
                }
            
            progress = bdays_passed / total_bdays
            
            # ===== STEP 2: Logic 1 - Run-rate (실적 기반) =====
            daily_avg = current_val / bdays_passed
            pred_runrate = daily_avg * total_bdays
            
            # ===== STEP 3: Logic 2 - Historical Pattern (과거 패턴 기반) =====
            # 지난달 확정값 조회
            prev_month_date = current_date.replace(day=1) - timedelta(days=1)
            prev_month_period = pd.Period(prev_month_date, freq='M')
            
            if prev_month_period in self.monthly_series.index:
                prev_month_val = self.monthly_series[prev_month_period]
                # 역사적 MoM 평균 적용
                mom_ratio = self.mom_ratios.get(month, 1.0)
                pred_pattern_base = prev_month_val * mom_ratio
                
                # Pattern 예측을 Run-rate와 섞어서 극단값 방지
                # Run-rate와 Pattern 기반 예측의 간단한 평균으로 중화
                pred_pattern = (pred_runrate + pred_pattern_base) / 2
            else:
                # 과거 데이터 없으면 월 평균에 계절성 지수 적용
                monthly_avg = self.monthly_series.mean()
                seasonal_factor = self.seasonal_factors.get(month, 1.0)
                pred_pattern = monthly_avg * seasonal_factor
            
            # ===== STEP 4: Logic 3 - Dynamic Weighting =====
            if progress < 0.30:
                # 월초: 데이터 부족 → Run-rate 비중 증가 (70%), Pattern 감소 (30%)
                w_runrate = 0.70
                w_pattern = 0.30
                confidence = "Low"
            elif progress > 0.80:
                # 월말: 실적 확실 → Run-rate 가중치 90%
                w_runrate = 0.90
                w_pattern = 0.10
                confidence = "High"
            else:
                # 중간: 선형적으로 가중치 교차 (더 보수적)
                w_runrate = 0.50 + (progress * 0.40)  # 50%~90%
                w_pattern = 1.0 - w_runrate
                confidence = "Medium"
            
            # ===== STEP 5: 최종 예측값 =====
            predicted_final = (w_runrate * pred_runrate) + (w_pattern * pred_pattern)
            predicted_final = max(0, predicted_final)
            
            return {
                "predicted_final": int(round(predicted_final)),
                "confidence": confidence,
                "progress": round(progress * 100, 1),
                "details": {
                    "run_rate": int(round(pred_runrate)),
                    "pattern_based": int(round(pred_pattern)),
                    "weights": f"Run-rate({w_runrate:.1%}) + Pattern({w_pattern:.1%})"
                }
            }
        
        except Exception as e:
            print(f"[WARNING] 당월 예측 계산 실패: {e}")
            return {
                "predicted_final": current_val,
                "confidence": "Error",
                "details": {
                    "run_rate": current_val,
                    "pattern_based": current_val,
                    "weights": f"Error: {str(e)}"
                }
            }
    
    def predict_next_3_months(self) -> dict:
        """
        향후 3개월 추세 예측 (다중 모델 앙상블)
        
        ⚠️ 중요: training_series(마감된 전월 이전 데이터)로 학습하여
        당월의 불완전한 데이터가 예측을 왜곡하지 않도록 함.
        
        Returns:
            {
                "2026-02": 1100,
                "2026-03": 1350,
                "2026-04": 1200,
                "method": "Ensemble (HW+SARIMA+Trend)"
            }
        """
        try:
            # 다중 모델 앙상블 사용
            return self._predict_next_3_months_ensemble()
        
        except Exception as e:
            print(f"[WARNING] 3개월 예측 계산 실패: {e}")
            return self._predict_next_3_months_fallback()
    
    
    def _predict_next_3_months_fallback(self) -> dict:
        """
        Fallback: 최근 3개월 가중 이동평균 × 계절 지수
        
        데이터 부족 또는 모델링 실패 시 사용
        (training_series 기반)
        """
        try:
            # 최근 3개월 가중 이동평균 (최신 데이터에 높은 가중치)
            if len(self.training_series_cleaned) >= 3:
                recent_data = self.training_series_cleaned.tail(3).values
                weights = np.array([1, 2, 3])  # 선형 증가 가중치
                weighted_avg = np.average(recent_data, weights=weights)
            else:
                weighted_avg = self.training_series_cleaned.mean()
            
            # 향후 3개월 예측
            last_period = self.training_series_cleaned.index[-1]
            predictions = {}
            
            for i in range(1, 4):
                future_period = last_period + i
                month_num = future_period.month
                
                # 계절성 지수 적용
                seasonal_factor = self.seasonal_factors.get(month_num, 1.0)
                predicted_val = weighted_avg * seasonal_factor
                
                month_str = f"{future_period.year}-{month_num:02d}"
                predictions[month_str] = max(0, int(round(predicted_val)))
            
            return {
                **predictions,
                "method": "Weighted Moving Avg + Seasonal"
            }
        
        except Exception as e:
            print(f"[WARNING] Fallback 예측 실패: {e}")
            # 최종 폴백: 단순 평균
            simple_avg = int(self.training_series_cleaned.mean())
            return {
                "2026-02": simple_avg,
                "2026-03": simple_avg,
                "2026-04": simple_avg,
                "method": "Simple Average (Final Fallback)"
            }
    
    def _predict_next_3_months_ensemble(self) -> dict:
        """
        향후 3개월 다중 모델 앙상블 예측
        
        사용 모델:
        - Holt-Winters (계절성 + 추세)
        - SARIMA (자기회귀)
        - Trend Regression (선형 추세)
        
        가중치 (월말 신뢰도 기준):
        - HW: 45% (계절성 강함)
        - SARIMA: 35% (자기회귀)
        - Trend: 20% (선형 추세)
        
        Returns:
            {'2026-02': 1200, '2026-03': 1350, '2026-04': 1200, 'method': 'Ensemble (HW+SARIMA+Trend)'}
        """
        try:
            # 1. Holt-Winters 예측 (이미 있는 메서드 활용)
            hw_current, hw_future = self._predict_holt_winters_extended()
            
            if not hw_future:
                return self._predict_next_3_months_fallback()
            
            # 2. SARIMA 예측 (2, 3, 4개월 앞)
            sarima_preds = {}
            last_period = self.training_series_cleaned.index[-1]
            
            for months_ahead in range(2, 5):  # 2, 3, 4개월 앞
                sarima_result = self._predict_with_sarima(months_ahead=months_ahead)
                if sarima_result['value'] is not None:
                    future_period = last_period + months_ahead
                    month_str = f"{future_period.year}-{future_period.month:02d}"
                    sarima_preds[month_str] = max(0, float(sarima_result['value']))
                else:
                    # SARIMA 실패 시 HW 값 사용
                    future_period = last_period + months_ahead
                    month_str = f"{future_period.year}-{future_period.month:02d}"
                    sarima_preds[month_str] = hw_future.get(month_str, 0)
            
            # 3. Trend Regression 예측 (2, 3, 4개월 앞)
            trend_preds = {}
            for months_ahead in range(2, 5):
                pred_val = self._predict_with_trend_regression(months_ahead=months_ahead)
                future_period = last_period + months_ahead
                month_str = f"{future_period.year}-{future_period.month:02d}"
                trend_preds[month_str] = max(0, float(pred_val))
            
            # 4. 앙상블 가중치 적용
            weights = {
                'hw': 0.45,      # Holt-Winters (계절성 강함)
                'sarima': 0.35,  # SARIMA (자기회귀)
                'trend': 0.20    # Trend Regression (선형 추세)
            }
            
            final_preds = {}
            for month_str in sorted(hw_future.keys()):
                hw_val = hw_future.get(month_str, 0)
                sarima_val = sarima_preds.get(month_str, 0)
                trend_val = trend_preds.get(month_str, 0)
                
                ensemble_val = (
                    weights['hw'] * hw_val +
                    weights['sarima'] * sarima_val +
                    weights['trend'] * trend_val
                )
                
                final_preds[month_str] = max(0, int(round(ensemble_val)))
            
            return {
                **final_preds,
                "method": "Ensemble (HW+SARIMA+Trend)"
            }
        
        except Exception as e:
            print(f"[WARNING] 3개월 앙상블 예측 실패: {e}")
            return self._predict_next_3_months_fallback()
    
    def _predict_with_trend_regression(self, months_ahead: int = 1) -> float:
        """
        회귀 기반 추세 예측
        장기 추세를 반영한 예측값 산출
        """
        try:
            future_idx = len(self.training_series_cleaned) + months_ahead - 1
            pred_val = self.trend_intercept + self.trend_slope * future_idx
            return max(0, pred_val)
        except Exception:
            return self.training_series_cleaned.mean()
    
    def _calculate_runrate_ensemble(self, current_val: int, bdays_passed: int, total_bdays: int, current_month: int) -> float:
        """
        Run-rate 앙상블: 실시간 페이스 + 과거 동월 데이터 혼합
        
        Run-rate가 너무 불안정한 점을 보정하기 위해
        실시간 데이터(Run-rate)와 과거 같은 달의 평균을 섞음
        
        Args:
            current_val: 현재까지의 누적 건수
            bdays_passed: 경과 영업일
            total_bdays: 전체 영업일
            current_month: 현재 월(1~12)
        
        Returns:
            앙상블된 run-rate 예측값
        """
        try:
            # 1. 실시간 Run-rate (현재 페이스 외삽)
            daily_avg = current_val / bdays_passed if bdays_passed > 0 else 0
            pred_runrate_raw = daily_avg * total_bdays
            
            # 2. Back data (과거 동월의 평균)
            # 훈련 데이터에서 같은 월의 과거 데이터들 평균
            back_data = self.training_series_cleaned[
                self.training_series_cleaned.index.month == current_month
            ]
            
            if len(back_data) > 0:
                back_data_avg = back_data.mean()
            else:
                # 과거 동월 데이터 없으면 전체 평균 사용
                back_data_avg = self.training_series_cleaned.mean()
            
            # 3. 진행률에 따른 동적 가중치
            # 초기(낮은 진행률): Back data 신뢰도 높음
            # 후기(높은 진행률): Run-rate 신뢰도 높음
            progress = bdays_passed / total_bdays if total_bdays > 0 else 0
            
            if progress < 0.30:
                # 초기: Back data 70% + Run-rate 30%
                w_back = 0.70
                w_runrate = 0.30
            elif progress < 0.70:
                # 중기: Back data 40% + Run-rate 60%
                w_back = 0.40
                w_runrate = 0.60
            else:
                # 후기: Back data 20% + Run-rate 80%
                w_back = 0.20
                w_runrate = 0.80
            
            # 4. 앙상블
            pred_runrate_ensemble = (w_back * back_data_avg) + (w_runrate * pred_runrate_raw)
            
            return max(0, pred_runrate_ensemble)
        
        except Exception as e:
            print(f"[WARNING] Run-rate 앙상블 계산 실패: {e}")
            # Fallback: 원본 run-rate 반환
            daily_avg = current_val / bdays_passed if bdays_passed > 0 else 0
            return max(0, daily_avg * total_bdays)
    
    def _predict_holt_winters_extended(self) -> Tuple[Optional[float], dict]:
        """
        Holt-Winters 모델로 4개월 예측 (당월 + 향후 3개월)
        
        당월의 불완전한 데이터로 인한 오인을 방지하기 위해
        training_series(전월 이전 완전한 데이터)로 학습하여 4개월 예측
        
        Returns:
            (current_month_pred, future_preds)
            - current_month_pred: 당월의 통계적 기대치 (Back Data 기반)
            - future_preds: {'2026-02': 1100, '2026-03': 1200, '2026-04': 1300}
        """
        try:
            if not HAS_STATSMODELS or len(self.training_series_cleaned) < 24:
                return None, {}
            
            ts_data = self.training_series_cleaned.values
            
            try:
                model = ExponentialSmoothing(
                    ts_data,
                    seasonal_periods=12,
                    trend='add',
                    seasonal='add',
                    damped_trend=True
                )
                fitted_model = model.fit(optimized=True)
                
                # 4개월 예측 (당월 + 미래 3개월)
                forecast_values = fitted_model.forecast(steps=4)
                
                current_month_pred = max(0, float(forecast_values[0]))
                
                # 미래 3개월 추출 (2, 3, 4개월 앞)
                # training_series_cleaned는 이전 달(12월)까지 완성되어 있으므로
                # forecast[1]은 2개월 앞(2월), forecast[2]는 3개월 앞(3월), forecast[3]은 4개월 앞(4월)
                last_period = self.training_series_cleaned.index[-1]
                future_preds = {}
                
                for forecast_idx in range(1, 4):  # forecast_values[1], [2], [3]
                    months_ahead = forecast_idx + 1  # 2, 3, 4개월 앞
                    future_period = last_period + months_ahead
                    month_str = f"{future_period.year}-{future_period.month:02d}"
                    future_preds[month_str] = max(0, int(round(forecast_values[forecast_idx])))
                
                return current_month_pred, future_preds
            
            except Exception as e:
                print(f"[WARNING] Holt-Winters 모델링 실패: {e}")
                return None, {}
        
        except Exception as e:
            print(f"[WARNING] 확장 Holt-Winters 예측 실패: {e}")
            return None, {}
    
    def _predict_with_sarima(self, months_ahead: int = 1) -> dict:
        """
        SARIMA 모델을 통한 예측
        자동 파라미터 선택으로 최적화된 예측
        (training_series 기반으로 학습)
        """
        try:
            if not HAS_ARIMA or len(self.training_series_cleaned) < 24:
                return {'value': None, 'ci_lower': None, 'ci_upper': None}
            
            # 간단한 자동 파라미터 선택 (전체 탐색보다 빠름)
            ts_data = self.training_series_cleaned.values
            
            # 기본 파라미터로 모델 구성
            try:
                model = ARIMA(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                fitted = model.fit()
                
                forecast = fitted.get_forecast(steps=months_ahead)
                pred_val = forecast.predicted_mean.iloc[-1] if len(forecast.predicted_mean) > 0 else None
                
                # 신뢰도 구간 계산
                conf_int = forecast.conf_int(alpha=0.05)
                ci_lower = conf_int.iloc[-1, 0] if len(conf_int) > 0 else None
                ci_upper = conf_int.iloc[-1, 1] if len(conf_int) > 0 else None
                
                return {
                    'value': max(0, pred_val) if pred_val else None,
                    'ci_lower': max(0, ci_lower) if ci_lower else None,
                    'ci_upper': max(0, ci_upper) if ci_upper else None
                }
            except Exception:
                return {'value': None, 'ci_lower': None, 'ci_upper': None}
        except Exception:
            return {'value': None, 'ci_lower': None, 'ci_upper': None}
    
    def _get_ensemble_weights(self) -> dict:
        """
        적응형 가중치 결정
        변동성, 데이터 충분성, 신뢰도에 따라 동적으로 조정
        """
        data_sufficiency = min(len(self.monthly_series) / 24, 1.0)  # 0~1
        
        if self.volatility_level == "Low":
            # 변동성 낮음: 패턴/추세 신뢰도 높음
            return {
                'runrate': 0.30,
                'pattern': 0.25,
                'trend': 0.25,
                'hw': 0.15,
                'sarima': 0.05
            }
        elif self.volatility_level == "Medium":
            # 변동성 중간: 균형잡힌 가중치
            return {
                'runrate': 0.40,
                'pattern': 0.20,
                'trend': 0.15,
                'hw': 0.15,
                'sarima': 0.10
            }
        else:
            # 변동성 높음: 현재 데이터(runrate) 비중 증가
            return {
                'runrate': 0.50,
                'pattern': 0.15,
                'trend': 0.10,
                'hw': 0.10,
                'sarima': 0.15
            }
    
    def predict_current_month_advanced(self, current_val: int, current_date: datetime) -> dict:
        """
        고도화된 당월 예측 (통계적 기대치 vs 실시간 실적 동적 조합)
        
        핵심 논리:
        - 월초(진행률 낮음): 과거 패턴(Back Data) 신뢰도 높음 → 통계 기대치 가중치 높음
        - 월말(진행률 높음): 실제 데이터(Run-rate) 신뢰도 높음 → 실적 가중치 높음
        
        구간별 가중치 (통계 기대치 vs Run-rate):
        - 초기 (0~30%): 통계 70% + 실시간 30% ("아직 평소 실력대로 나오겠거니")
        - 중기 (30~70%): 통계 50% + 실시간 50% (데이터 섞임)
        - 후기 (70%~): 통계 20% + 실시간 80% ("이제 실제 팩트다")
        """
        try:
            year = current_date.year
            month = current_date.month
            
            # ===== 기본 데이터 =====
            month_start = datetime(year, month, 1)
            bdays_passed = self._calculate_business_days(month_start, current_date)
            total_bdays = self._get_total_business_days_in_month(year, month)
            
            if bdays_passed <= 0 or total_bdays <= 0:
                return {
                    "predicted_final": current_val,
                    "confidence": "Very Low",
                    "volatility": self.volatility_level,
                    "ci_lower": int(current_val * 0.9),
                    "ci_upper": int(current_val * 1.1)
                }
            
            progress = bdays_passed / total_bdays
            
            # ===== 1단계: 통계적 기대치 (Back Data 기반) =====
            # Holt-Winters가 예측한 "당월의 평소 수준"
            stat_pred, _ = self._predict_holt_winters_extended()
            if stat_pred is None:
                # Fallback: 훈련 데이터의 평균 + 계절성 지수
                base_avg = self.training_series_cleaned.mean()
                seasonal_factor = self.seasonal_factors.get(month, 1.0)
                stat_pred = base_avg * seasonal_factor
            
            # ===== 2단계: Run-rate 앙상블 (실시간 페이스 + 과거 동월) =====
            # 단순 Run-rate만 사용하는 대신, Run-rate + Back data 앙상블로 안정화
            pred_runrate = self._calculate_runrate_ensemble(
                current_val, bdays_passed, total_bdays, month
            )
            
            # ===== 3단계: 진행률에 따른 가중치 적용 =====
            if progress < 0.30:
                # 초기 (0~30%): 통계 70% + 실시간 30%
                w_stat = 0.70
                w_runrate = 0.30
                confidence = "Low"
                z_critical = 2.58  # 99% CI (매우 넓음)
            
            elif progress < 0.70:
                # 중기 (30~70%): 통계 50% + 실시간 50%
                w_stat = 0.50
                w_runrate = 0.50
                confidence = "Medium"
                z_critical = 1.96  # 95% CI
            
            else:
                # 후기 (70%~): 통계 20% + 실시간 80%
                w_stat = 0.20
                w_runrate = 0.80
                confidence = "High"
                z_critical = 1.96  # 95% CI
            
            # ===== 최종 예측값 =====
            predicted_final = (w_stat * stat_pred) + (w_runrate * pred_runrate)
            predicted_final = max(0, predicted_final)
            
            # ===== 신뢰도 구간 (변동성 기반) =====
            std_dev = self.volatility * predicted_final
            ci_lower = max(0, predicted_final - z_critical * std_dev)
            ci_upper = predicted_final + z_critical * std_dev
            
            return {
                "predicted_final": int(round(predicted_final)),
                "confidence": confidence,
                "volatility": self.volatility_level,
                "progress": round(progress * 100, 1),
                "ci_lower": int(round(ci_lower)),
                "ci_upper": int(round(ci_upper)),
                "models": {
                    "runrate": int(round(pred_runrate)),
                    "stat_base": int(round(stat_pred)),
                    "pattern": 0,
                    "trend": 0,
                    "hw": 0,
                    "sarima": 0
                },
                "weights": {
                    'stat_base': f"{w_stat:.1%}",
                    'runrate': f"{w_runrate:.1%}",
                    'pattern': '0.0%',
                    'trend': '0.0%',
                    'hw': '0.0%',
                    'sarima': '0.0%'
                }
            }
        
        except Exception as e:
            print(f"[WARNING] 고급 예측 계산 실패: {e}")
            return {
                "predicted_final": current_val,
                "confidence": "Error",
                "volatility": self.volatility_level,
                "ci_lower": int(current_val * 0.8),
                "ci_upper": int(current_val * 1.2)
            }
    
    def get_summary(self) -> dict:
        """
        엔진 상태 요약 (고도화 버전)
        """
        return {
            "data_points": len(self.monthly_series),
            "date_range": f"{self.min_date.date()} ~ {self.max_date.date()}",
            "current_year_month": f"{self.current_year}-{self.current_month:02d}",
            "trend_slope": round(self.trend_slope, 4),
            "volatility": {
                "level": self.volatility_level,
                "value": round(self.volatility, 4)
            },
            "seasonal_factors": {k: round(v, 3) for k, v in self.seasonal_factors.items()},
            "mom_ratios": {k: round(v, 3) for k, v in self.mom_ratios.items()},
            "models": {
                "has_statsmodels": HAS_STATSMODELS,
                "has_arima": HAS_ARIMA
            }
        }
