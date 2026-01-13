"""
🧪 Simulation Lab Engine (Track B - v6.0 Dynamic Ensemble)
==========================================================
Logic:
1. Validation Phase: Hide last 3 months -> Train -> Predict -> Calculate MAE.
2. Weighting: Calculate dynamic weights based on Inverse Error (1/MAE).
3. Final Phase: Retrain on full data -> Forecast Future -> Apply Weights.
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
        
        # 전체 데이터 (Final 학습용)
        self.full_data = self.ts
        
        # [FIX] UI 호환성을 위해 'train_data' 속성 추가 (전체 데이터와 동일하게 매핑)
        self.train_data = self.ts 
        
        # 검증용 데이터 분할 (최근 3개월을 검증셋으로 사용)
        if len(self.ts) > 12:
            self.train_val = self.ts.iloc[:-3]  # 학습용 (검증 단계)
            self.test_val = self.ts.iloc[-3:]   # 평가용 (검증 단계)
        else:
            # 데이터가 너무 적으면 분할 없이 진행
            self.train_val = self.ts
            self.test_val = self.ts.iloc[-1:]

        self.model_weights = {} # 계산된 가중치 저장
    # =========================================================================
    # [Core] Individual Model Runners (Generic)
    # =========================================================================
    
    def _run_prophet_internal(self, train_data, periods) -> List[float]:
        if not HAS_PROPHET: return []
        try:
            df_p = train_data.reset_index()
            df_p.columns = ['ds', 'y']
            m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
            m.fit(df_p)
            future = m.make_future_dataframe(periods=periods, freq='M')
            fcst = m.predict(future)
            return [max(0, x) for x in fcst.tail(periods)['yhat'].values]
        except:
            return []

    def _run_automl_internal(self, train_data, periods) -> List[float]:
        if not HAS_ML: return []
        try:
            # Feature Engineering
            df = pd.DataFrame(train_data)
            df.columns = ['y']
            for lag in [1, 2, 3, 12]: df[f'lag_{lag}'] = df['y'].shift(lag)
            df = df.dropna()
            
            if len(df) < 5: return [] # 데이터 부족
            
            X = df.drop(columns=['y'])
            y = df['y']
            
            # Optuna (Fast Mode)
            def objective(trial):
                param = {
                    'objective': 'regression', 'metric': 'mae', 'verbosity': -1,
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 40),
                }
                # Simple Split for speed
                split = int(len(X) * 0.8)
                model = lgb.LGBMRegressor(**param)
                model.fit(X.iloc[:split], y.iloc[:split])
                preds = model.predict(X.iloc[split:])
                return mean_absolute_error(y.iloc[split:], preds)

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=10, timeout=3)
            
            best_model = lgb.LGBMRegressor(**study.best_params)
            best_model.fit(X, y)
            
            # Recursive Forecast
            curr_ts = train_data.copy()
            preds = []
            for _ in range(periods):
                tmp = pd.DataFrame({'y': curr_ts})
                idx = curr_ts.index[-1] + pd.DateOffset(months=1)
                tmp.loc[idx] = 0
                for lag in [1, 2, 3, 12]: tmp[f'lag_{lag}'] = tmp['y'].shift(lag)
                feat = tmp.iloc[[-1]].drop(columns=['y'])
                val = max(0, best_model.predict(feat)[0])
                preds.append(val)
                curr_ts.loc[idx] = val
            return preds
        except:
            return []

    def _run_sarima_internal(self, train_data, periods) -> List[float]:
        if not HAS_STATS: return []
        try:
            # Robust Order (1,1,1)x(1,1,0,12)
            model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,0,12))
            fit = model.fit(disp=False)
            return [max(0, x) for x in fit.forecast(steps=periods)]
        except:
            return []

    # =========================================================================
    # [Logic] Dynamic Weight Calculation (Backtesting)
    # =========================================================================
    def _calculate_weights(self) -> Dict[str, float]:
        """
        검증셋(Hold-out)을 통해 각 모델의 MAE를 계산하고, 역수 가중치를 산출함.
        """
        errors = {}
        val_len = len(self.test_val)
        
        # 1. Validation Run
        p_pred = self._run_prophet_internal(self.train_val, val_len)
        m_pred = self._run_automl_internal(self.train_val, val_len)
        s_pred = self._run_sarima_internal(self.train_val, val_len)
        
        y_true = self.test_val.values
        
        # 2. Calculate MAE
        if p_pred and len(p_pred) == val_len:
            errors['Prophet'] = mean_absolute_error(y_true, p_pred)
        else:
            errors['Prophet'] = float('inf') # 실패 시 무한대 에러
            
        if m_pred and len(m_pred) == val_len:
            errors['AutoML'] = mean_absolute_error(y_true, m_pred)
        else:
            errors['AutoML'] = float('inf')
            
        if s_pred and len(s_pred) == val_len:
            errors['SARIMAX'] = mean_absolute_error(y_true, s_pred)
        else:
            errors['SARIMAX'] = float('inf')
            
        # 3. Inverse Weighting (에러가 작을수록 가중치 큼)
        # Weight = (1/MAE) / Sum(1/MAE)
        inverse_errors = {}
        for k, v in errors.items():
            if v == 0: v = 1e-6 # 0 나누기 방지
            if v == float('inf'):
                inverse_errors[k] = 0
            else:
                inverse_errors[k] = 1 / v
                
        total_inv = sum(inverse_errors.values())
        
        weights = {}
        if total_inv == 0:
            # 모두 실패했으면 균등 배분
            weights = {'Prophet': 0.33, 'AutoML': 0.33, 'SARIMAX': 0.33}
        else:
            for k, v in inverse_errors.items():
                weights[k] = v / total_inv
                
        return weights

    # =========================================================================
    # [Main] Competition & Ensemble
    # =========================================================================
    def run_competition(self, periods=4) -> pd.DataFrame:
        """
        1. Backtest로 가중치 계산
        2. 전체 데이터로 재학습 & 예측
        3. 앙상블 적용
        """
        # 1. Calculate Dynamic Weights
        self.model_weights = self._calculate_weights()
        
        # 2. Final Forecast (Retrain on Full Data)
        p_final = self._run_prophet_internal(self.full_data, periods)
        m_final = self._run_automl_internal(self.full_data, periods)
        s_final = self._run_sarima_internal(self.full_data, periods)
        
        # 3. Organize DataFrame
        last_date = self.full_data.index[-1]
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods)]
        
        result_df = pd.DataFrame({'Date': future_dates}).set_index('Date')
        
        # 각 모델 결과 담기
        if p_final: result_df['Prophet'] = p_final
        if m_final: result_df['AutoML'] = m_final
        if s_final: result_df['SARIMAX'] = s_final
        
        # 4. Apply Ensemble
        # 가중치 적용하여 최종 'Ensemble' 컬럼 생성
        ensemble_vals = np.zeros(periods)
        valid_weight_sum = 0
        
        for model_name, weight in self.model_weights.items():
            if model_name in result_df.columns:
                ensemble_vals += result_df[model_name].values * weight
                valid_weight_sum += weight
        
        # 결과 정규화 (혹시 모델 하나가 실패해서 가중치 합이 1이 안 될 경우 대비)
        if valid_weight_sum > 0:
            result_df['Ensemble'] = ensemble_vals / valid_weight_sum
        else:
            # 모든 모델 실패 시 0
            result_df['Ensemble'] = 0
            
        return result_df

    # =========================================================================
    # [Allocation] Top-down using Ensemble
    # =========================================================================
    def predict_with_allocation(self, plant, major_category, sub_df, periods=3, forecast_df=None) -> pd.DataFrame:
        """
        앙상블 결과('Ensemble')를 사용하여 하위 배분 수행
        """
        if forecast_df is None or 'Ensemble' not in forecast_df.columns:
            return pd.DataFrame()
            
        future_preds = forecast_df['Ensemble'].values
        
        if len(future_preds) == 0: 
            return pd.DataFrame()
        
        future_dates = forecast_df.index
        allocation_results = []
        
        for date_obj, total_pred in zip(future_dates, future_preds):
            target_month = date_obj.month
            
            # Ratio Calculation
            if not pd.api.types.is_datetime64_any_dtype(sub_df['접수일자']):
                sub_df['접수일자'] = pd.to_datetime(sub_df['접수일자'])
                
            history_same_month = sub_df[sub_df['접수일자'].dt.month == target_month]
            if history_same_month.empty:
                recent = sub_df['접수일자'].max() - pd.DateOffset(months=3)
                history_same_month = sub_df[sub_df['접수일자'] >= recent]
                
            sub_agg = history_same_month.groupby('소분류')['건수'].sum().reset_index()
            total_hist = sub_agg['건수'].sum()
            
            if total_hist > 0: sub_agg['ratio'] = sub_agg['건수'] / total_hist
            else: sub_agg['ratio'] = 1.0 / len(sub_agg) if len(sub_agg) > 0 else 0
                
            for _, row in sub_agg.iterrows():
                allocation_results.append({
                    '플랜트': plant,
                    '대분류': major_category,
                    '소분류': row['소분류'],
                    '예측월': date_obj.strftime('%Y-%m'),
                    '예측건수': round(total_pred * row['ratio'], 1),
                    '점유율': f"{row['ratio']:.1%}"
                })
                
        return pd.DataFrame(allocation_results)