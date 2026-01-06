# ============================================================================
# Optuna 기반 하이퍼파라미터 튜닝 및 챔피언 선정
# ============================================================================
# 설명: HyperParameterTuner로 3개 모델 최적화
#      ChampionSelector로 우승 모델 선정 및 예측

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error

from core.engine.models import SARIMAXModel, CatBoostModel, LSTMModel, BaseModel


# ============================================================================
# 1. 하이퍼파라미터 튜닝 (Optuna)
# ============================================================================

class HyperParameterTuner:
    """
    Optuna를 사용한 자동 하이퍼파라미터 튜닝.
    
    동작:
    1. 데이터를 Train(~3개월 전) / Test(마지막 3개월)로 분할
    2. 각 모델별 하이퍼파라미터 Search Space 정의
    3. Optuna로 최대 N 트라이얼 실행
    4. Test RMSE를 목표로 최적화
    """
    
    def __init__(
        self,
        n_trials: int = 20,
        test_months: int = 3,
        random_state: int = 42
    ):
        """
        튜너 초기화.
        
        Args:
            n_trials: Optuna 시행 횟수
            test_months: Test Set 기간 (개월)
            random_state: 난수 시드
        """
        self.n_trials = n_trials
        self.test_months = test_months
        self.random_state = random_state
        self.best_models = {}  # {model_name: best_model}
        self.best_params = {}  # {model_name: best_params}
        self.study_results = {}  # {model_name: study}
    
    def split_data(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.Series, pd.Series, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Train / Test 분할.
        
        동작:
        - 마지막 test_months를 Test로 분할
        - 나머지를 Train으로 사용
        
        Args:
            y: 시계열 데이터
            exog: 외생변수 (optional)
        
        Returns:
            Tuple: (y_train, y_test, exog_train, exog_test)
        """
        split_idx = len(y) - self.test_months
        
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        exog_train = None
        exog_test = None
        
        if exog is not None:
            exog_train = exog.iloc[:split_idx]
            exog_test = exog.iloc[split_idx:]
        
        print(f"[TUNER] 데이터 분할: Train {len(y_train)}개, Test {len(y_test)}개")
        return y_train, y_test, exog_train, exog_test
    
    def tune_sarimax(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
        exog_test: Optional[pd.DataFrame] = None
    ) -> Dict[str, any]:
        """
        SARIMAX 하이퍼파라미터 튜닝.
        
        Search Space:
        - order (p, d, q): (0-2, 0-2, 0-2)
        - seasonal_order (P, D, Q, s): (0-2, 0-1, 0-2, 12)
        """
        
        def objective(trial):
            try:
                # 하이퍼파라미터 제안
                p = trial.suggest_int('p', 0, 2)
                d = trial.suggest_int('d', 0, 2)
                q = trial.suggest_int('q', 0, 2)
                P = trial.suggest_int('P', 0, 2)
                D = trial.suggest_int('D', 0, 1)
                Q = trial.suggest_int('Q', 0, 2)
                
                # 모델 학습
                model = SARIMAXModel(
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, 12)
                )
                model.fit(None, y_train, exog=exog_train)
                
                # Test 예측
                forecast = model.predict(len(y_test), exog=exog_test)
                
                # RMSE 계산
                rmse = np.sqrt(mean_squared_error(y_test.values, forecast))
                
                return rmse
            
            except Exception as e:
                print(f"  [SARIMAX] 시행 실패: {str(e)}")
                return float('inf')
        
        # Optuna Study 실행
        print(f"[TUNER] SARIMAX 튜닝 시작 ({self.n_trials} trials)...")
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        self.study_results['SARIMAX'] = study
        best_params = study.best_params
        
        print(f"[TUNER] SARIMAX 최적 파라미터: {best_params}")
        print(f"[TUNER] SARIMAX 최적 RMSE: {study.best_value:.2f}")
        
        return best_params
    
    def tune_catboost(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
        exog_test: Optional[pd.DataFrame] = None
    ) -> Dict[str, any]:
        """
        CatBoost 하이퍼파라미터 튜닝.
        
        Search Space:
        - lag_features: 1-6
        - iterations: 50-500
        """
        
        def objective(trial):
            try:
                # 하이퍼파라미터 제안
                lag_features = trial.suggest_int('lag_features', 1, 6)
                iterations = trial.suggest_int('iterations', 50, 500, step=50)
                
                # 모델 학습
                model = CatBoostModel(
                    lag_features=lag_features,
                    iterations=iterations
                )
                model.fit(None, y_train, exog=exog_train)
                
                # Test 예측
                forecast = model.predict(len(y_test), exog=exog_test)
                
                # RMSE 계산
                rmse = np.sqrt(mean_squared_error(y_test.values, forecast))
                
                return rmse
            
            except Exception as e:
                print(f"  [CatBoost] 시행 실패: {str(e)}")
                return float('inf')
        
        # Optuna Study 실행
        print(f"[TUNER] CatBoost 튜닝 시작 ({self.n_trials} trials)...")
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        self.study_results['CatBoost'] = study
        best_params = study.best_params
        
        print(f"[TUNER] CatBoost 최적 파라미터: {best_params}")
        print(f"[TUNER] CatBoost 최적 RMSE: {study.best_value:.2f}")
        
        return best_params
    
    def tune_lstm(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
        exog_test: Optional[pd.DataFrame] = None
    ) -> Dict[str, any]:
        """
        LSTM 하이퍼파라미터 튜닝.
        
        Search Space:
        - lookback: 6-24
        - hidden_size: 32-256
        - epochs: 50-200
        """
        
        def objective(trial):
            try:
                # 하이퍼파라미터 제안
                lookback = trial.suggest_int('lookback', 6, 24)
                hidden_size = trial.suggest_int('hidden_size', 32, 256, step=32)
                epochs = trial.suggest_int('epochs', 50, 200, step=50)
                
                # 모델 학습
                model = LSTMModel(
                    lookback=lookback,
                    hidden_size=hidden_size,
                    epochs=epochs,
                    batch_size=8
                )
                model.fit(None, y_train, exog=exog_train)
                
                # Test 예측
                forecast = model.predict(len(y_test), exog=exog_test)
                
                # RMSE 계산
                rmse = np.sqrt(mean_squared_error(y_test.values, forecast))
                
                return rmse
            
            except Exception as e:
                print(f"  [LSTM] 시행 실패: {str(e)}")
                return float('inf')
        
        # Optuna Study 실행
        print(f"[TUNER] LSTM 튜닝 시작 ({self.n_trials} trials)...")
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        self.study_results['LSTM'] = study
        best_params = study.best_params
        
        print(f"[TUNER] LSTM 최적 파라미터: {best_params}")
        print(f"[TUNER] LSTM 최적 RMSE: {study.best_value:.2f}")
        
        return best_params
    
    def tune_all(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame] = None
    ) -> Dict[str, any]:
        """
        3개 모델 모두 튜닝.
        
        Args:
            y: 시계열 데이터
            exog: 외생변수
        
        Returns:
            Dict: {model_name: best_params}
        """
        # 데이터 분할
        y_train, y_test, exog_train, exog_test = self.split_data(y, exog)
        
        # 각 모델 튜닝
        sarimax_params = self.tune_sarimax(y_train, y_test, exog_train, exog_test)
        catboost_params = self.tune_catboost(y_train, y_test, exog_train, exog_test)
        lstm_params = self.tune_lstm(y_train, y_test, exog_train, exog_test)
        
        self.best_params = {
            'SARIMAX': sarimax_params,
            'CatBoost': catboost_params,
            'LSTM': lstm_params
        }
        
        return self.best_params


# ============================================================================
# 2. 챔피언 모델 선정 및 예측
# ============================================================================

class ChampionSelector:
    """
    3개 모델의 성능 비교 및 우승 모델 선정.
    
    동작:
    1. 최적 파라미터로 3개 모델 재학습
    2. Test RMSE 비교 → 우승 모델 선정
    3. 우승 모델로 향후 6개월 예측
    """
    
    def __init__(self, best_params: Dict[str, any]):
        """
        선정자 초기화.
        
        Args:
            best_params: {model_name: best_params}
        """
        self.best_params = best_params
        self.models = {}
        self.leaderboard = None
        self.champion = None
        self.champion_name = None
    
    def train_models(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        test_months: int = 3
    ) -> pd.DataFrame:
        """
        최적 파라미터로 3개 모델 학습.
        
        Args:
            y: 시계열 데이터
            exog: 외생변수
            test_months: Test 기간 (개월)
        
        Returns:
            pd.DataFrame: 성능 리더보드
        """
        # Train / Test 분할
        split_idx = len(y) - test_months
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        exog_train = None
        exog_test = None
        if exog is not None:
            exog_train = exog.iloc[:split_idx]
            exog_test = exog.iloc[split_idx:]
        
        results = []
        
        # 1. SARIMAX
        try:
            print("[CHAMPION] SARIMAX 모델 학습...")
            params = self.best_params.get('SARIMAX', {})
            sarimax = SARIMAXModel(
                order=(params.get('p', 1), params.get('d', 1), params.get('q', 1)),
                seasonal_order=(params.get('P', 1), params.get('D', 1), params.get('Q', 1), 12)
            )
            sarimax.fit(None, y_train, exog=exog_train)
            
            forecast_sarimax = sarimax.predict(len(y_test), exog=exog_test)
            rmse_sarimax = np.sqrt(mean_squared_error(y_test.values, forecast_sarimax))
            
            self.models['SARIMAX'] = sarimax
            results.append({
                'Model': 'SARIMAX',
                'RMSE': rmse_sarimax,
                'Rank': 0
            })
            print(f"[CHAMPION] SARIMAX RMSE: {rmse_sarimax:.2f}")
        
        except Exception as e:
            print(f"[CHAMPION] SARIMAX 학습 실패: {str(e)}")
            results.append({'Model': 'SARIMAX', 'RMSE': float('inf'), 'Rank': 0})
        
        # 2. CatBoost
        try:
            print("[CHAMPION] CatBoost 모델 학습...")
            params = self.best_params.get('CatBoost', {})
            catboost = CatBoostModel(
                lag_features=params.get('lag_features', 3),
                iterations=params.get('iterations', 100)
            )
            catboost.fit(None, y_train, exog=exog_train)
            
            forecast_catboost = catboost.predict(len(y_test), exog=exog_test)
            rmse_catboost = np.sqrt(mean_squared_error(y_test.values, forecast_catboost))
            
            self.models['CatBoost'] = catboost
            results.append({
                'Model': 'CatBoost',
                'RMSE': rmse_catboost,
                'Rank': 0
            })
            print(f"[CHAMPION] CatBoost RMSE: {rmse_catboost:.2f}")
        
        except Exception as e:
            print(f"[CHAMPION] CatBoost 학습 실패: {str(e)}")
            results.append({'Model': 'CatBoost', 'RMSE': float('inf'), 'Rank': 0})
        
        # 3. LSTM
        try:
            print("[CHAMPION] LSTM 모델 학습...")
            params = self.best_params.get('LSTM', {})
            lstm = LSTMModel(
                lookback=params.get('lookback', 12),
                hidden_size=params.get('hidden_size', 64),
                epochs=params.get('epochs', 100),
                batch_size=8
            )
            lstm.fit(None, y_train, exog=exog_train)
            
            forecast_lstm = lstm.predict(len(y_test), exog=exog_test)
            rmse_lstm = np.sqrt(mean_squared_error(y_test.values, forecast_lstm))
            
            self.models['LSTM'] = lstm
            results.append({
                'Model': 'LSTM',
                'RMSE': rmse_lstm,
                'Rank': 0
            })
            print(f"[CHAMPION] LSTM RMSE: {rmse_lstm:.2f}")
        
        except Exception as e:
            print(f"[CHAMPION] LSTM 학습 실패: {str(e)}")
            results.append({'Model': 'LSTM', 'RMSE': float('inf'), 'Rank': 0})
        
        # 리더보드 생성
        self.leaderboard = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
        self.leaderboard['Rank'] = range(1, len(self.leaderboard) + 1)
        
        # 챔피언 선정
        champion_row = self.leaderboard.iloc[0]
        self.champion_name = champion_row['Model']
        self.champion = self.models[self.champion_name]
        
        print(f"\n🏆 [CHAMPION] 우승 모델: {self.champion_name} (RMSE: {champion_row['RMSE']:.2f})")
        
        return self.leaderboard
    
    def forecast(self, y: pd.Series, exog: Optional[pd.DataFrame] = None, steps: int = 6) -> np.ndarray:
        """
        챔피언 모델로 미래 예측.
        
        Args:
            y: 전체 시계열 (재학습 기반)
            exog: 미래 외생변수
            steps: 예측 단계 수 (개월)
        
        Returns:
            np.ndarray: 예측값
        """
        if self.champion is None:
            raise ValueError("챔피언 모델이 선정되지 않음")
        
        # 전체 데이터로 재학습
        print(f"[CHAMPION] {self.champion_name}로 최종 학습...")
        self.champion.fit(None, y, exog=exog)
        
        # 예측
        forecast = self.champion.predict(steps, exog=exog)
        
        return forecast
    
    def get_leaderboard(self) -> pd.DataFrame:
        """리더보드 반환."""
        return self.leaderboard.copy()
    
    def get_champion_info(self) -> Dict[str, any]:
        """챔피언 모델 정보."""
        if self.leaderboard is not None:
            return self.leaderboard.iloc[0].to_dict()
        return {}
