"""
🧪 Simulation Lab Engine (Track B - v6.1 Zero-Trend Guard)
==========================================================
Logic:
1. Dead Check: If recent history is flat zero, skip training and predict 0.
2. Validation Phase: Hide last 3 months -> Train -> Predict -> Calculate MAE.
3. Weighting: Calculate dynamic weights based on Inverse Error (1/MAE).
4. Final Phase: Retrain on full data -> Forecast Future -> Apply Weights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings

# --- Dependency Check ---
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_ML = True
except ImportError:
    HAS_ML = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATS = True
except ImportError:
    HAS_STATS = False

warnings.filterwarnings('ignore')

class SimulationEngine:
    def __init__(self, df: pd.DataFrame, date_col='접수일자', val_col='건수'):
        self.raw_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.raw_df[date_col]):
            self.raw_df[date_col] = pd.to_datetime(self.raw_df[date_col])
            
        # 월별 집계
        self.ts = self.raw_df.set_index(date_col).resample('M')[val_col].sum().asfreq('M', fill_value=0)
        
        # [FIX] 미마감 당월 제외 (노이즈 방지)
        last_date = self.raw_df[date_col].max()
        last_day_of_month = pd.Timestamp(last_date.year, last_date.month, 1) + pd.offsets.MonthEnd(0)
        is_month_complete = (last_date.date() == last_day_of_month.date())
        
        if not is_month_complete and len(self.ts) > 0:
            # 미마감 당월 제외
            self.full_data = self.ts.iloc[:-1]
            self.current_partial = self.ts.iloc[-1]  # 당월 부분 데이터 (참고용)
        else:
            self.full_data = self.ts
            self.current_partial = 0
        
        self.train_data = self.full_data  # UI 호환용
        
        # 검증용 데이터 분할 (최근 3개월)
        if len(self.full_data) > 12:
            self.train_val = self.full_data.iloc[:-3]
            self.test_val = self.full_data.iloc[-3:]
        else:
            self.train_val = self.full_data
            self.test_val = self.full_data.iloc[-1:] if len(self.full_data) > 0 else self.full_data

        self.model_weights = {} 
        
        # [NEW] 소멸(Dead) 판정
        # 최근 12개월 중 0건인 달이 10개월 이상이거나, 최근 6개월 연속 0건이면 Dead로 간주
        self.is_dead = self._check_dead_signal(self.full_data)

    def _check_dead_signal(self, data: pd.Series) -> bool:
        """최근 데이터가 소멸 추세인지 확인"""
        if len(data) < 6: return False
        
        # Rule 1: 최근 6개월 연속 0건
        recent_6m = data.tail(6)
        if recent_6m.sum() == 0:
            return True
            
        # Rule 2: 최근 12개월 중 90%가 0건이고 합계가 5건 미만 (간헐적 발생도 무시)
        if len(data) >= 12:
            recent_12m = data.tail(12)
            zero_count = (recent_12m == 0).sum()
            total_sum = recent_12m.sum()
            if zero_count >= 10 and total_sum < 5:
                return True
                
        return False

    # =========================================================================
    # [Core] Individual Model Runners
    # =========================================================================
    
    def _run_prophet_internal(self, train_data, periods) -> List[float]:
        if not HAS_PROPHET: return []
        # Dead Guard
        if self._check_dead_signal(train_data): return [0.0] * periods
        
        try:
            df_p = train_data.reset_index()
            df_p.columns = ['ds', 'y']
            
            # Cap floor at 0 (Logistic growth/decay requires cap/floor, linear doesn't but we force max(0))
            m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
            m.fit(df_p)
            future = m.make_future_dataframe(periods=periods, freq='M')
            fcst = m.predict(future)
            preds = fcst.tail(periods)['yhat'].values
            return [max(0, x) for x in preds]
        except:
            return []

    def _run_automl_internal(self, train_data, periods) -> List[float]:
        if not HAS_ML: return []
        # Dead Guard
        if self._check_dead_signal(train_data): return [0.0] * periods
        
        try:
            df = pd.DataFrame({'y': train_data})
            for lag in [1, 2, 3, 6, 12]:
                df[f'lag_{lag}'] = df['y'].shift(lag)
            df = df.dropna()
            
            if len(df) < 5: return []
            
            X = df.drop(columns=['y'])
            y = df['y']
            
            def objective(trial):
                param = {
                    'objective': 'regression', 'metric': 'mae', 'verbosity': -1,
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 40),
                }
                split = int(len(X) * 0.8)
                model = lgb.LGBMRegressor(**param)
                model.fit(X.iloc[:split], y.iloc[:split])
                preds = model.predict(X.iloc[split:])
                return mean_absolute_error(y.iloc[split:], preds)

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=10, timeout=3)
            
            best_model = lgb.LGBMRegressor(**study.best_params)
            best_model.fit(X, y)
            
            curr_ts = train_data.copy()
            preds = []
            for _ in range(periods):
                idx = curr_ts.index[-1] + pd.DateOffset(months=1)
                tmp = pd.DataFrame({'y': curr_ts})
                tmp.loc[idx] = 0
                for lag in [1, 2, 3, 6, 12]: 
                    tmp[f'lag_{lag}'] = tmp['y'].shift(lag)
                
                feat = tmp.iloc[[-1]].drop(columns=['y'])
                val = max(0, best_model.predict(feat)[0])
                preds.append(val)
                curr_ts = pd.concat([curr_ts, pd.Series([val], index=[idx])])
                
            return preds
        except:
            return []

    def _run_sarima_internal(self, train_data, periods) -> List[float]:
        if not HAS_STATS: return []
        # Dead Guard
        if self._check_dead_signal(train_data): return [0.0] * periods
        
        try:
            # Enforce stationarity check or simple model if data is sparse
            if (train_data == 0).mean() > 0.5:
                # 데이터 절반 이상이 0이면 SARIMA 수렴 어려움 -> 0 반환
                return [0.0] * periods
                
            model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,0,12))
            fit = model.fit(disp=False)
            return [max(0, x) for x in fit.forecast(steps=periods)]
        except:
            return []

    # =========================================================================
    # [Logic] Dynamic Weight Calculation (Backtesting)
    # =========================================================================
    def _calculate_weights(self) -> Dict[str, float]:
        if self.is_dead:
            return {'Prophet': 0.33, 'AutoML': 0.33, 'SARIMAX': 0.33}
            
        errors = {}
        val_len = len(self.test_val)
        
        p_pred = self._run_prophet_internal(self.train_val, val_len)
        m_pred = self._run_automl_internal(self.train_val, val_len)
        s_pred = self._run_sarima_internal(self.train_val, val_len)
        
        y_true = self.test_val.values
        
        for name, pred in [('Prophet', p_pred), ('AutoML', m_pred), ('SARIMAX', s_pred)]:
            if pred and len(pred) == val_len:
                errors[name] = mean_absolute_error(y_true, pred)
            else:
                errors[name] = float('inf')
                
        inverse_errors = {}
        for k, v in errors.items():
            if v == 0: v = 1e-6
            if v == float('inf'): inverse_errors[k] = 0
            else: inverse_errors[k] = 1 / v
            
        total_inv = sum(inverse_errors.values())
        weights = {}
        if total_inv == 0:
            weights = {'Prophet': 0.33, 'AutoML': 0.33, 'SARIMAX': 0.33}
        else:
            for k, v in inverse_errors.items():
                weights[k] = v / total_inv
                
        return weights

    # =========================================================================
    # [Main] Competition & Ensemble
    # =========================================================================
    def run_competition(self, periods=4) -> pd.DataFrame:
        last_date = self.full_data.index[-1]
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods)]
        result_df = pd.DataFrame({'Date': future_dates}).set_index('Date')

        # 1. Check Dead Signal First
        if self.is_dead:
            result_df['Prophet'] = 0.0
            result_df['AutoML'] = 0.0
            result_df['SARIMAX'] = 0.0
            result_df['Ensemble'] = 0.0
            self.model_weights = {'Prophet': 0.33, 'AutoML': 0.33, 'SARIMAX': 0.33} # Dummy
            return result_df

        # 2. Normal Process
        self.model_weights = self._calculate_weights()
        
        p_final = self._run_prophet_internal(self.full_data, periods)
        m_final = self._run_automl_internal(self.full_data, periods)
        s_final = self._run_sarima_internal(self.full_data, periods)
        
        if p_final: result_df['Prophet'] = p_final
        if m_final: result_df['AutoML'] = m_final
        if s_final: result_df['SARIMAX'] = s_final
        
        ensemble_vals = np.zeros(periods)
        valid_weight_sum = 0
        
        for model_name, weight in self.model_weights.items():
            if model_name in result_df.columns:
                ensemble_vals += result_df[model_name].values * weight
                valid_weight_sum += weight
                
        if valid_weight_sum > 0:
            result_df['Ensemble'] = ensemble_vals / valid_weight_sum
        else:
            result_df['Ensemble'] = 0
            
        return result_df

    # =========================================================================
    # [Allocation] Time-Weighted Distribution (4_예측_시뮬레이션과 로직 통일)
    # =========================================================================
    def predict_with_allocation(self, plant, major_category, sub_df, periods=3, forecast_df=None) -> pd.DataFrame:
        """시간 가중 분배 로직 (지수 감쇠 + 소멸 추세 감지)"""
        if forecast_df is None or 'Ensemble' not in forecast_df.columns:
            return pd.DataFrame()
            
        future_preds = forecast_df['Ensemble'].values
        if len(future_preds) == 0 or sub_df.empty:
            return pd.DataFrame()
        
        # 소분류별 시간 가중 비율 계산
        if not pd.api.types.is_datetime64_any_dtype(sub_df['접수일자']):
            sub_df['접수일자'] = pd.to_datetime(sub_df['접수일자'])
        
        # 월별 데이터 생성
        df_monthly = sub_df.copy()
        df_monthly['년월'] = df_monthly['접수일자'].dt.to_period('M')
        
        # 소분류 × 년월 피벗
        monthly_pivot = pd.pivot_table(
            df_monthly,
            index='소분류',
            columns='년월',
            values='건수',
            aggfunc='sum',
            fill_value=0
        )
        
        if monthly_pivot.empty:
            return pd.DataFrame()
        
        # 시간 가중치 계산 (지수 감쇠)
        n_months = len(monthly_pivot.columns)
        decay_rate = 0.92
        time_weights = np.array([decay_rate ** (n_months - i - 1) for i in range(n_months)])
        time_weights = time_weights / time_weights.sum()
        
        # 각 소분류별 가중 평균 계산
        weighted_totals = (monthly_pivot.values * time_weights).sum(axis=1)
        weighted_series = pd.Series(weighted_totals, index=monthly_pivot.index)
        
        # 소멸 추세 감지
        recent_12m = monthly_pivot.iloc[:, -12:] if monthly_pivot.shape[1] >= 12 else monthly_pivot
        recent_avg = recent_12m.mean(axis=1)
        historical_avg = monthly_pivot.mean(axis=1)
        extinction_ratio = recent_avg / historical_avg.replace(0, 1)
        is_extinct = extinction_ratio < 0.2
        
        # 가중치 조정: 소멸 추세면 최근 데이터만 사용
        final_ratios = weighted_series.copy()
        for idx in weighted_series.index:
            if is_extinct.loc[idx]:
                recent_6m = recent_12m.loc[idx].tail(6)
                final_ratios.loc[idx] = recent_6m.mean()
        
        # 정규화
        total_sum = final_ratios.sum()
        if total_sum > 0:
            ratios = final_ratios / total_sum
        else:
            # 모든 소분류가 0이면 균등 분배
            ratios = pd.Series(1.0 / len(final_ratios), index=final_ratios.index)
        
        # 예측 결과 생성
        allocation_results = []
        future_dates = forecast_df.index
        
        for date_obj, total_pred in zip(future_dates, future_preds):
            for sub_category, ratio in ratios.items():
                if ratio > 0:  # 비율이 0보다 큰 경우만
                    allocation_results.append({
                        '플랜트': plant,
                        '대분류': major_category,
                        '소분류': sub_category,
                        '예측월': date_obj.strftime('%Y-%m'),
                        '예측건수': round(total_pred * ratio, 1),
                        '점유율': f"{ratio:.1%}"
                    })
                
        return pd.DataFrame(allocation_results)