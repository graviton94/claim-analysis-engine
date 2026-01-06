# ============================================================================
# ML/DL 모델 엔진: CatBoost, SARIMAX, LSTM
# ============================================================================
# 설명: 시계열 예측을 위한 3개 모델 구현
#      BaseModel 추상 클래스를 상속하여 fit/predict 표준화

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================================
# 기본 추상 클래스
# ============================================================================

class BaseModel(ABC):
    """
    모든 예측 모델의 기본 추상 클래스.
    
    모든 모델은 다음 메서드를 구현해야 함:
    - fit(X, y, exog=None): 모델 학습
    - predict(steps, exog=None): 미래 예측
    """
    
    def __init__(self, name: str):
        """
        모델 초기화.
        
        Args:
            name: 모델 이름
        """
        self.name = name
        self.is_fitted = False
        self.train_rmse = None
        self.test_rmse = None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        모델 학습.
        
        Args:
            X: 입력 특징
            y: 타겟 값
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        미래 값 예측.
        
        Args:
            steps: 예측 단계 수
        
        Returns:
            np.ndarray: 예측값
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """모델 정보 반환."""
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'train_rmse': self.train_rmse,
            'test_rmse': self.test_rmse
        }


# ============================================================================
# 1. SARIMAX 모델
# ============================================================================

class SARIMAXModel(BaseModel):
    """
    SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables).
    
    특징:
    - 시계열의 계절성 반영
    - 매출수량을 외생변수(exog)로 사용
    - RMSE 기반 성능 평가
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)
    ):
        """
        SARIMAX 모델 초기화.
        
        Args:
            order: (p, d, q) - AR, 차분, MA 차수
            seasonal_order: (P, D, Q, s) - 계절성 파라미터
        """
        super().__init__("SARIMAX")
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None
        self.training_data = None
        self.exog_train = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        SARIMAX 모델 학습.
        
        Args:
            X: 미사용 (호환성)
            y: 타겟 시계열 (예: 월별 클레임 건수)
            exog: 외생변수 (예: 매출수량)
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        try:
            # 모델 생성 및 학습
            self.model = SARIMAX(
                y,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.results = self.model.fit(disp=False, maxiter=200)
            
            # 훈련 데이터 저장
            self.training_data = y.copy()
            self.exog_train = exog.copy() if exog is not None else None
            
            self.is_fitted = True
            print(f"[{self.name}] 모델 학습 완료")
        
        except Exception as e:
            print(f"[{self.name}] 학습 실패: {str(e)}")
            raise
    
    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        미래 값 예측.
        
        Args:
            steps: 예측 단계 수 (월 수)
            exog: 미래 외생변수 (필수)
        
        Returns:
            np.ndarray: 예측값
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name}: 모델이 학습되지 않음")
        
        if exog is None:
            raise ValueError(f"{self.name}: 외생변수(매출수량) 필수")
        
        try:
            forecast = self.results.get_forecast(
                steps=steps,
                exog=exog
            )
            return forecast.predicted_mean.values
        
        except Exception as e:
            print(f"[{self.name}] 예측 실패: {str(e)}")
            raise


# ============================================================================
# 2. CatBoost 모델
# ============================================================================

class CatBoostModel(BaseModel):
    """
    CatBoost (Gradient Boosting Decision Tree).
    
    특징:
    - Lag Feature 자동 생성 (t-1, t-2, t-3)
    - 매출수량 추가 피처
    - 빠른 학습 및 높은 정확도
    """
    
    def __init__(self, lag_features: int = 3, iterations: int = 100):
        """
        CatBoost 모델 초기화.
        
        Args:
            lag_features: Lag 특징 개수 (1, 2, 3개월 전)
            iterations: Boosting 반복 횟수
        """
        super().__init__("CatBoost")
        self.lag_features = lag_features
        self.iterations = iterations
        self.model = None
        self.scaler = None
        self.training_y = None
        self.training_exog = None
    
    def _create_lag_features(self, y: pd.Series) -> pd.DataFrame:
        """
        Lag Feature 생성.
        
        Args:
            y: 시계열 데이터
        
        Returns:
            pd.DataFrame: Lag Feature 포함 데이터프레임
        """
        df = pd.DataFrame({'y': y})
        
        for i in range(1, self.lag_features + 1):
            df[f'lag_{i}'] = df['y'].shift(i)
        
        return df.dropna()
    
    def fit(self, X: pd.DataFrame, y: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        CatBoost 모델 학습.
        
        Args:
            X: 미사용
            y: 타겟 시계열
            exog: 외생변수 (매출수량)
        """
        from catboost import CatBoostRegressor
        from sklearn.preprocessing import StandardScaler
        
        try:
            # Lag Feature 생성
            lag_df = self._create_lag_features(y)
            X_features = lag_df.drop('y', axis=1)
            y_train = lag_df['y']
            
            # 외생변수 추가
            if exog is not None:
                exog_aligned = exog.iloc[len(y) - len(y_train):].reset_index(drop=True)
                X_features['sales'] = exog_aligned['매출수량'].values if '매출수량' in exog.columns else exog_aligned.iloc[:, 0].values
            
            # 모델 학습
            self.model = CatBoostRegressor(
                iterations=self.iterations,
                verbose=False,
                random_state=42
            )
            self.model.fit(X_features, y_train)
            
            # 훈련 데이터 저장
            self.training_y = y.copy()
            self.training_exog = exog.copy() if exog is not None else None
            
            self.is_fitted = True
            print(f"[{self.name}] 모델 학습 완료")
        
        except Exception as e:
            print(f"[{self.name}] 학습 실패: {str(e)}")
            raise
    
    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        미래 값 예측.
        
        Args:
            steps: 예측 단계 수
            exog: 미래 외생변수
        
        Returns:
            np.ndarray: 예측값
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name}: 모델이 학습되지 않음")
        
        try:
            predictions = []
            last_y = self.training_y.iloc[-self.lag_features:].values.tolist()
            
            for step in range(steps):
                # Lag Feature 구성
                features = {}
                for i in range(1, min(self.lag_features + 1, len(last_y) + 1)):
                    features[f'lag_{i}'] = last_y[-(i)]
                
                # 외생변수 추가
                if exog is not None and step < len(exog):
                    features['sales'] = exog['매출수량'].iloc[step] if '매출수량' in exog.columns else exog.iloc[step, 0]
                
                # 예측
                X_pred = pd.DataFrame([features])
                pred = self.model.predict(X_pred)[0]
                predictions.append(pred)
                last_y.append(pred)
            
            return np.array(predictions)
        
        except Exception as e:
            print(f"[{self.name}] 예측 실패: {str(e)}")
            raise


# ============================================================================
# 3. LSTM 모델
# ============================================================================

class LSTMModel(BaseModel):
    """
    LSTM (Long Short-Term Memory).
    
    특징:
    - PyTorch 기반 신경망
    - 시계열 Window Dataset 자동 생성
    - 배치 학습 지원
    """
    
    def __init__(
        self,
        lookback: int = 12,
        hidden_size: int = 64,
        epochs: int = 100,
        batch_size: int = 16
    ):
        """
        LSTM 모델 초기화.
        
        Args:
            lookback: 윈도우 크기 (과거 몇 개월)
            hidden_size: LSTM 은닉층 크기
            epochs: 학습 에포크 수
            batch_size: 배치 크기
        """
        super().__init__("LSTM")
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_y = None
        self.y_min = None
        self.y_max = None
    
    def _create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 윈도우 데이터셋 생성.
        
        Args:
            data: 시계열 데이터
            lookback: 윈도우 크기
        
        Returns:
            Tuple: (X, y) - 입력과 타겟
        """
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Min-Max 정규화."""
        self.y_min = data.min()
        self.y_max = data.max()
        return (data - self.y_min) / (self.y_max - self.y_min + 1e-8)
    
    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """역 정규화."""
        return data * (self.y_max - self.y_min + 1e-8) + self.y_min
    
    def fit(self, X: pd.DataFrame, y: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        LSTM 모델 학습.
        
        Args:
            X: 미사용
            y: 타겟 시계열
            exog: 미사용
        """
        try:
            # 데이터 정규화
            y_data = y.values.astype(np.float32)
            y_normalized = self._normalize(y_data)
            
            # 시계열 데이터셋 생성
            X_seq, y_seq = self._create_sequences(y_normalized, self.lookback)
            
            if len(X_seq) == 0:
                raise ValueError(f"{self.name}: 데이터 부족 (최소 {self.lookback + 1}개 필요)")
            
            # PyTorch 텐서 변환
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            y_tensor = torch.FloatTensor(y_seq).reshape(-1, 1).to(self.device)
            
            # 모델 정의
            class LSTMNet(nn.Module):
                def __init__(self, input_size, hidden_size, output_size=1):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    _, (h_n, _) = self.lstm(x)
                    out = self.fc(h_n[-1])
                    return out
            
            self.model = LSTMNet(1, self.hidden_size).to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # 학습
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            for epoch in range(self.epochs):
                for X_batch, y_batch in dataloader:
                    optimizer.zero_grad()
                    pred = self.model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()
            
            self.training_y = y.copy()
            self.is_fitted = True
            print(f"[{self.name}] 모델 학습 완료")
        
        except Exception as e:
            print(f"[{self.name}] 학습 실패: {str(e)}")
            raise
    
    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        미래 값 예측.
        
        Args:
            steps: 예측 단계 수
            exog: 미사용
        
        Returns:
            np.ndarray: 예측값
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name}: 모델이 학습되지 않음")
        
        try:
            # 마지막 lookback 시점의 정규화된 값
            y_normalized = self._normalize(self.training_y.values.astype(np.float32))
            last_seq = y_normalized[-self.lookback:].copy()
            
            predictions = []
            
            self.model.eval()
            with torch.no_grad():
                for _ in range(steps):
                    # 입력 시퀀스
                    X_input = torch.FloatTensor(last_seq.reshape(1, -1, 1)).to(self.device)
                    
                    # 예측
                    pred_normalized = self.model(X_input).item()
                    predictions.append(pred_normalized)
                    
                    # 시퀀스 업데이트
                    last_seq = np.append(last_seq[1:], pred_normalized)
            
            # 역 정규화
            predictions_denorm = self._denormalize(np.array(predictions))
            return np.maximum(predictions_denorm, 0)  # 음수 값 제거
        
        except Exception as e:
            print(f"[{self.name}] 예측 실패: {str(e)}")
            raise
