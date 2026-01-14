"""
📊 Advanced Forecasting Engine Module (v4.3 - Zero-Trend Guard)
====================================================================
Architecture: 3-Way Ensemble (Stabilizer + Learner + Judge) with Optuna
1. Auto-Tuning: Runs Optuna on every request to find best LightGBM params.
2. Recursive Forecasting: Predicts 4 months continuously.
3. Stabilizer (STL): Robust decomposition as a safety net.
4. [NEW] Zero-Trend Guard: Forces prediction to 0 if recent history is silent.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from typing import Dict, Tuple, List, Optional
import warnings

# --- Dependency Check ---
try:
    from statsmodels.tsa.seasonal import STL
    HAS_STL = True
except ImportError:
    HAS_STL = False

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING) # 로그 억제
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings('ignore')

class ForecastEngine:
    def __init__(self, daily_df: pd.DataFrame, date_col: str = '접수일자', value_col: str = '건수'):
        self.raw_df = daily_df.copy()
        
        # 날짜 컬럼 보정
        if not pd.api.types.is_datetime64_any_dtype(self.raw_df[date_col]):
            self.raw_df[date_col] = pd.to_datetime(self.raw_df[date_col])
            
        self.date_col = date_col
        self.val_col = value_col
        
        # 현재 시점 기준 (오늘)
        self.today = datetime.now()
        self.current_month_start = self.today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # 월별 집계 (전체 이력)
        # 1. 월별 그룹핑
        monthly_grp = self.raw_df.set_index(date_col).resample('M')[value_col].size()
        # 2. 인덱스를 월초(MS)로 변환하여 다루기 쉽게 함
        monthly_grp.index = monthly_grp.index.to_period('M').to_timestamp()
        
        # 3. 빈 달 채우기 (Zero-filling) - 매우 중요
        if not monthly_grp.empty:
            full_idx = pd.date_range(start=monthly_grp.index.min(), end=self.current_month_start, freq='MS')
            self.monthly_ts = monthly_grp.reindex(full_idx, fill_value=0)
        else:
            self.monthly_ts = pd.Series(dtype=float)

        # 당월 누적 실적 (Current Month Actual)
        mask_curr = (self.raw_df[date_col] >= self.current_month_start) & (self.raw_df[date_col] <= self.today)
        self.current_actual = len(self.raw_df[mask_curr])
        
        # 학습용 데이터 (당월 제외, 전월까지)
        if not self.monthly_ts.empty:
            self.train_ts = self.monthly_ts[self.monthly_ts.index < self.current_month_start]
        else:
            self.train_ts = pd.Series(dtype=float)

    # -------------------------------------------------------------------------
    # [Logic 1] Zero-Trend Guard (소멸 판정)
    # -------------------------------------------------------------------------
    def _check_recent_silence(self, months: int = 12) -> bool:
        """
        최근 N개월간 발생 실적이 0건이면 '소멸(Dead)'로 판정.
        과거에 100건이 발생했어도 최근 12개월간 0건이면 예측은 0이어야 함.
        """
        if len(self.train_ts) < months:
            return False # 데이터가 너무 짧으면 판단 유보
            
        recent_data = self.train_ts.tail(months)
        if recent_data.sum() == 0:
            return True
        return False

    # -------------------------------------------------------------------------
    # [Logic 2] Run-rate (진행률 기반 단순 추정)
    # -------------------------------------------------------------------------
    def _predict_run_rate(self) -> float:
        """
        단순 Run-rate: (현재실적 / 경과일) * 전체일수
        단, 월 초반(1~5일)에는 전월/전전월 평균 가중치 부여
        """
        days_in_month = (self.current_month_start + relativedelta(months=1) - timedelta(days=1)).day
        days_passed = self.today.day
        
        # 진행률
        progress = days_passed / days_in_month
        
        # A. 순수 Run-rate
        if days_passed > 0:
            simple_rr = self.current_actual / progress
        else:
            simple_rr = 0
            
        # B. 보정 로직 (월 초반 변동성 완화)
        # 최근 6개월 평균 (직전 6개월)
        if len(self.train_ts) >= 6:
            recent_avg = self.train_ts.tail(6).mean()
        else:
            recent_avg = simple_rr
            
        # 가중치: 월말에 가까울수록 simple_rr 비중 증가
        # day 1: 0% rr, day 15: 50% rr, day 30: 100% rr
        rr_weight = min(1.0, progress) 
        
        final_rr = (simple_rr * rr_weight) + (recent_avg * (1 - rr_weight))
        return max(self.current_actual, final_rr) # 적어도 현재 실적보단 커야 함

    # -------------------------------------------------------------------------
    # [Logic 3] STL Decomposition (계절성 반영)
    # -------------------------------------------------------------------------
    def _predict_stl(self, periods=4) -> List[float]:
        if not HAS_STL or len(self.train_ts) < 24: 
            # 데이터 부족 시 최근 3개월 평균으로 대체
            if len(self.train_ts) > 0:
                avg = self.train_ts.tail(3).mean()
                return [avg] * periods
            return [0.0] * periods
            
        try:
            # Robust STL
            stl = STL(self.train_ts, period=12, seasonal=13).fit()
            
            # Trend 추출 (최근 Trend 값 유지)
            last_trend = stl.trend.iloc[-1]
            
            # Seasonality 추출 (내년도 월별 계절성 복사)
            last_month = self.train_ts.index[-1].month
            future_seasonals = []
            
            for i in range(1, periods + 1):
                next_m = (last_month + i - 1) % 12 + 1
                # 과거 해당 월의 seasonal 성분 평균값 사용
                seasonal_comp = stl.seasonal[stl.seasonal.index.month == next_m].mean()
                future_seasonals.append(seasonal_comp)
                
            # Forecast = Trend + Seasonal
            forecasts = [max(0, last_trend + s) for s in future_seasonals]
            return forecasts
        except:
            return [self.train_ts.mean()] * periods

    # -------------------------------------------------------------------------
    # [Logic 4] LightGBM with Optuna (AutoML)
    # -------------------------------------------------------------------------
    def _predict_automl(self, periods=4) -> List[float]:
        if not HAS_LGBM or not HAS_OPTUNA or len(self.train_ts) < 12:
            return [0.0] * periods
            
        try:
            # Dataset Preparation (Lag Features)
            df = pd.DataFrame({'y': self.train_ts})
            for lag in [1, 2, 3, 6, 12]:
                df[f'lag_{lag}'] = df['y'].shift(lag)
            
            df = df.dropna()
            if len(df) < 10: return [0.0] * periods
            
            X = df.drop(columns=['y'])
            y = df['y']
            
            # Quick Optuna (제한 시간 2초)
            def objective(trial):
                param = {
                    'objective': 'regression', 'metric': 'rmse', 'verbosity': -1,
                    'n_estimators': trial.suggest_int('n_estimators', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 30),
                    'min_child_samples': trial.suggest_int('min_child_samples', 2, 10)
                }
                
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                for train_index, test_index in tscv.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    model = lgb.LGBMRegressor(**param)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    scores.append(np.mean((preds - y_test)**2))
                return np.mean(scores)

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=10, timeout=5) # 속도 최우선
            
            best_model = lgb.LGBMRegressor(**study.best_params)
            best_model.fit(X, y)
            
            # Recursive Forecast
            preds = []
            curr_ts = self.train_ts.copy()
            
            for _ in range(periods):
                # Feature Construction for next step
                last_idx = curr_ts.index[-1]
                next_idx = last_idx + pd.DateOffset(months=1)
                
                feat = {}
                for lag in [1, 2, 3, 6, 12]:
                    if len(curr_ts) >= lag:
                        feat[f'lag_{lag}'] = curr_ts.iloc[-lag]
                    else:
                        feat[f'lag_{lag}'] = 0
                
                feat_df = pd.DataFrame([feat])
                pred_val = max(0, best_model.predict(feat_df)[0])
                preds.append(pred_val)
                
                # Append prediction to history for recursion
                curr_ts = pd.concat([curr_ts, pd.Series([pred_val], index=[next_idx])])
                
            return preds
        except Exception as e:
            return [0.0] * periods

    # -------------------------------------------------------------------------
    # [Main] Forecast 4 Months
    # -------------------------------------------------------------------------
    def forecast_4m(self) -> Dict:
        """
        당월 포함 향후 4개월 예측 (Zero-Trend Guard 적용)
        """
        # 1. Zero-Trend Guard: 최근 6개월간 0건이면 예측도 0건 (강제 종료)
        if self._check_recent_silence(months=6):
            return {
                "current": {
                    "predicted_final": self.current_actual, # 현재까지 접수된 것만 인정
                    "current_actual": self.current_actual,
                    "progress_ratio": 1.0,
                    "details": {"trend_status": "🔴 소멸 (최근 6개월 0건)"}
                },
                "future_4m": {
                    (self.current_month_start + relativedelta(months=i)).strftime('%Y-%m'): 0 
                    for i in range(4)
                },
                "extinction_info": {"is_extinct": True}
            }
            
        # 2. Individual Forecasts
        pred_run = self._predict_run_rate() # 당월용
        preds_stl = self._predict_stl(periods=4)
        preds_lgbm = self._predict_automl(periods=4)
        
        # 3. Dynamic Weighting (당월)
        days_in_month = (self.current_month_start + relativedelta(months=1) - timedelta(days=1)).day
        progress = min(1.0, self.today.day / days_in_month)
        
        # 진행률이 높을수록 Run-rate 신뢰도 상승
        w_run = 0.4 + (progress * 0.4) # 0.4 ~ 0.8
        w_model = 1.0 - w_run          # 0.6 ~ 0.2
        
        model_avg_curr = (preds_stl[0] + preds_lgbm[0]) / 2
        final_current = (pred_run * w_run) + (model_avg_curr * w_model)
        final_current = max(self.current_actual, final_current) # 실적보다 낮을 순 없음
        
        # 4. Future Ensemble (익월 ~ 3개월 뒤)
        # 미래로 갈수록 LGBM(학습) 비중을 높임 (STL은 단순 패턴)
        future_map = {}
        dates = [self.current_month_start + relativedelta(months=i) for i in range(4)]
        
        # Month 0 (Current)
        future_map[dates[0].strftime('%Y-%m')] = int(final_current)
        
        # Month 1~3
        for i in range(1, 4):
            val = (preds_lgbm[i] * 0.6) + (preds_stl[i] * 0.4)
            
            # [Trend Decay] 미래 예측이 과거 평균보다 터무니없이 높으면 억제
            hist_avg = self.train_ts.tail(12).mean()
            if hist_avg > 0 and val > hist_avg * 3:
                val = (val + hist_avg * 3) / 2 # 댐핑
            
            future_map[dates[i].strftime('%Y-%m')] = int(val)
            
        return {
            "current": {
                "predicted_final": int(final_current),
                "current_actual": int(self.current_actual),
                "progress_ratio": progress,
                "details": {
                    "run_rate": int(pred_run),
                    "stl": int(preds_stl[0]),
                    "lgbm": int(preds_lgbm[0]),
                    "weights_desc": f"Run({w_run:.0%}) | AI({w_model:.0%})"
                }
            },
            "future_4m": future_map,
            "extinction_info": {"is_extinct": False}
        }