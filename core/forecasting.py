"""
📊 Advanced Forecasting Engine Module (v4.2 - Real-time Auto-Tuning)
====================================================================
Architecture: 3-Way Ensemble (Stabilizer + Learner + Judge) with Optuna
1. Auto-Tuning: Runs Optuna on every request to find best LightGBM params.
2. Recursive Forecasting: Predicts 4 months continuously.
3. Stabilizer (STL): Robust decomposition as a safety net.
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
        if not pd.api.types.is_datetime64_any_dtype(self.raw_df[date_col]):
            self.raw_df[date_col] = pd.to_datetime(self.raw_df[date_col])
            
        # 1. Monthly Aggregation
        self.monthly_series = self.raw_df.set_index(date_col).resample('M')[value_col].sum()
        self.monthly_series = self.monthly_series.asfreq('M', fill_value=0)
        self.last_date = self.raw_df[date_col].max()
        
        # 2. Data Guard
        days_in_month = self.last_date.days_in_month
        is_month_end = (self.last_date.day == days_in_month)
        
        if not is_month_end and len(self.monthly_series) > 0:
            self.train_series = self.monthly_series.iloc[:-1] 
            self.current_actual = self.monthly_series.iloc[-1]
        else:
            self.train_series = self.monthly_series
            self.current_actual = 0 if self.monthly_series.empty else self.monthly_series.iloc[-1]

    def _get_business_days_progress(self) -> Tuple[float, int, int]:
        """영업일 기준 진행률"""
        today = self.last_date
        year, month = today.year, today.month
        start_date = date(year, month, 1)
        end_date = start_date + relativedelta(day=31)
        try:
            total_biz = np.busday_count(start_date, end_date + timedelta(days=1))
            passed_biz = np.busday_count(start_date, today.date() + timedelta(days=1))
            if total_biz == 0: return 1.0, 1, 1
            progress = passed_biz / total_biz
            return min(max(progress, 0.05), 1.0), passed_biz, total_biz
        except:
            return today.day / today.days_in_month, today.day, today.days_in_month

    # =========================================================================
    # [Module 1] The Stabilizer: STL
    # =========================================================================
    def _predict_stl(self, steps=4) -> List[float]:
        if not HAS_STL or len(self.train_series) < 24:
            avg = self.train_series.mean() if len(self.train_series) > 0 else 0
            return [avg] * steps

        try:
            res = STL(self.train_series, period=12, robust=True).fit()
            last_trend = res.trend.iloc[-1]
            future_preds = []
            
            last_date = self.train_series.index[-1]
            for i in range(1, steps + 1):
                target_date = last_date + relativedelta(months=i)
                seasonal_comp = res.seasonal[res.seasonal.index.month == target_date.month].mean()
                pred = last_trend + seasonal_comp
                future_preds.append(max(0, float(pred)))
            return future_preds
        except:
            return [self.train_series.mean()] * steps

    # =========================================================================
    # [Module 2] The Learner: LightGBM with Optuna Auto-Tuning
    # =========================================================================
    def _create_features(self, series: pd.Series, lags=[1, 2, 3, 12], window=3) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.DataFrame(series.copy())
        df.columns = ['y']
        for lag in lags:
            df[f'lag_{lag}'] = df['y'].shift(lag)
        df[f'rolling_mean_{window}'] = df['y'].shift(1).rolling(window=window).mean()
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        df = df.dropna()
        return df.drop(columns=['y']), df['y']

    def _optimize_lgbm_params(self, X, y) -> Dict:
        """
        [NEW] Optuna를 이용한 실시간 하이퍼파라미터 최적화
        목표: 검증 오차(MAE)가 가장 낮은 파라미터 찾기
        """
        def objective(trial):
            param = {
                'objective': 'regression',
                'metric': 'mae',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 10, 40),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 20)
            }
            
            # TimeSeriesSplit (n_splits=3)
            # 과거 데이터를 3개 구간으로 나눠서 검증 -> 가장 일반화 잘 되는 파라미터 선정
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
                y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
                
                model = lgb.LGBMRegressor(**param)
                model.fit(X_t, y_t)
                preds = model.predict(X_v)
                scores.append(np.mean(np.abs(y_v - preds))) # MAE
            
            return np.mean(scores)

        # Create Study
        study = optuna.create_study(direction='minimize')
        # Timeout 3초: 대시보드 응답성을 위한 최소한의 타협
        study.optimize(objective, n_trials=15, timeout=3) 
        return study.best_params

    def _predict_lgbm_recursive(self, steps=4, first_step_value=None) -> List[float]:
        """Auto-Tuned LightGBM Recursive Forecast"""
        if not HAS_LGBM or len(self.train_series) < 24:
            return [0.0] * steps
            
        try:
            X_train, y_train = self._create_features(self.train_series)
            
            # 1. Parameter Selection
            if HAS_OPTUNA:
                try:
                    best_params = self._optimize_lgbm_params(X_train, y_train)
                    # 필수 고정 파라미터 병합
                    best_params.update({'objective': 'regression', 'verbosity': -1})
                    model = lgb.LGBMRegressor(**best_params)
                except:
                    # Optuna 실패 시 Fallback
                    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05)
            else:
                model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05)
            
            # 2. Train
            model.fit(X_train, y_train)
            
            # 3. Recursive Predict
            current_series = self.train_series.copy()
            future_preds = []
            
            for i in range(steps):
                if i == 0 and first_step_value is not None:
                    pred_val = first_step_value
                else:
                    # Feature Construction
                    temp_df = pd.DataFrame({'y': current_series})
                    next_date = current_series.index[-1] + relativedelta(months=1)
                    temp_df.loc[next_date] = 0
                    
                    for lag in [1, 2, 3, 12]:
                        temp_df[f'lag_{lag}'] = temp_df['y'].shift(lag)
                    temp_df[f'rolling_mean_3'] = temp_df['y'].shift(1).rolling(3).mean()
                    temp_df['month_sin'] = np.sin(2 * np.pi * temp_df.index.month / 12)
                    temp_df['month_cos'] = np.cos(2 * np.pi * temp_df.index.month / 12)
                    
                    X_next = temp_df.iloc[[-1]].drop(columns=['y'])
                    pred_val = model.predict(X_next)[0]
                    pred_val = max(0, float(pred_val))
                
                future_preds.append(pred_val)
                next_date = current_series.index[-1] + relativedelta(months=1)
                current_series.loc[next_date] = pred_val
                
            return future_preds
        except Exception as e:
            # print(f"LGBM Error: {e}")
            return [0.0] * steps

    # =========================================================================
    # [Module 3] The Judge: Unified 4M Pipeline
    # =========================================================================
    def forecast_4m(self) -> Dict:
        """
        [Main API] 당월(1월) + 미래(2,3,4월)을 한 번에 통합 예측.
        """
        progress, biz_passed, biz_total = self._get_business_days_progress()
        
        # --- Step 1. 당월(N월) 정밀 예측 ---
        pred_run = (self.current_actual / biz_passed) * biz_total if biz_passed > 0 else 0
        
        stl_first = self._predict_stl(steps=1)[0]
        # 첫 번째 스텝 예측 (Optuna 적용됨)
        lgbm_first_raw = self._predict_lgbm_recursive(steps=1)[0]
        if lgbm_first_raw == 0: lgbm_first_raw = stl_first
        
        # Divergence Guard
        if lgbm_first_raw > stl_first * 1.3: 
            lgbm_first = (lgbm_first_raw + stl_first) / 2
        else:
            lgbm_first = lgbm_first_raw
            
        # Weighting
        weights = {}
        if progress < 0.2:   weights = {'run': 0.1, 'model': 0.9}
        elif progress < 0.6: weights = {'run': 0.5, 'model': 0.5}
        else:                weights = {'run': 0.9, 'model': 0.1}
        
        model_val = (stl_first * 0.5) + (lgbm_first * 0.5)
        final_current = (pred_run * weights['run']) + (model_val * weights['model'])
        
        # --- Step 2. 미래(N+1, N+2, N+3) 연속 예측 ---
        # 1월 확정값(final_current)을 넘겨서 2,3,4월 예측
        lgbm_future_4m = self._predict_lgbm_recursive(steps=4, first_step_value=final_current)
        stl_future_4m = self._predict_stl(steps=4)
        
        result_map = {}
        last_date = self.train_series.index[-1]
        
        for i in range(4):
            target_date = last_date + relativedelta(months=i+1)
            date_str = target_date.strftime("%Y-%m")
            
            if i == 0:
                val = final_current
            else:
                val_lgbm = lgbm_future_4m[i]
                val_stl = stl_future_4m[i]
                
                if val_lgbm > val_stl * 1.3:
                    val = (val_lgbm * 0.4) + (val_stl * 0.6)
                else:
                    val = (val_lgbm * 0.7) + (val_stl * 0.3)
            
            result_map[date_str] = int(val)
            
        return {
            "current": {
                "predicted_final": int(final_current),
                "current_actual": int(self.current_actual),
                "progress_ratio": progress,
                "details": {
                    "run_rate": int(pred_run),
                    "stl": int(stl_first),
                    "lgbm": int(lgbm_first),
                    "weights_desc": f"Run({weights['run']:.0%}) | Optuna-ML({weights['model']:.0%})"
                }
            },
            "future_4m": result_map
        }