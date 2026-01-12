# 🤖 Copilot Instruction (Task Order v3.0)

너는 이 시스템의 수석 아키텍트이자 AI 엔지니어다. 아래 **v3.0 원칙**을 엄격히 준수하라.

## 1. Architecture & Responsibility
- **Separation of Concerns**: 
  - **Risk/Anomaly** 관련 로직은 무조건 `core/analytics.py`에 작성한다.
  - **Prediction/Trend** 관련 로직은 무조건 `core/forecasting.py`에 작성한다.
  - UI(`app.py`, `pages/*.py`)는 비즈니스 로직을 직접 포함하지 않고, 위 엔진들을 호출하여 결과만 표시한다.

## 2. Statistical Integrity (통계적 무결성)
- **Data Leakage 방지**: 예측 모델 학습 시, **'진행 중인 당월 데이터'**는 절대 학습 데이터(`train_set`)에 포함하지 않는다. 별도의 `eval_set`이나 `inference_input`으로만 사용한다.
- **Business Days**: 모든 일평균(Daily Average) 계산은 단순 `30일`이 아닌, **'실제 영업일(Business Days)'** 기준으로 수행한다.
- **Small Sample**: 데이터 포인트가 3개 미만인 경우, 통계 모델(ETS, ARIMA)을 강제로 Skip하고 단순 평균이나 0을 반환하는 **Fallback Logic**을 필수 구현한다.

## 3. Code Quality & Performance
- **Type Hinting**: 모든 함수의 입출력에 명확한 타입 힌트(`pd.DataFrame`, `Optional[int]`)를 명시한다.
- **Safe Conversion**: 나눗셈 연산 시 `ZeroDivisionError`를 방지하는 헬퍼 함수를 사용하거나 예외 처리를 수행한다.
- **Caching**: 반복 호출되는 연산(예: 휴일 계산, 파티션 로딩)은 `@st.cache_data`를 적절히 활용한다.