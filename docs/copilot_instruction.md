# 🤖 Copilot Instruction (Task Order v3.0)

너는 이 시스템의 수석 아키텍트다. 아래 **v3.0 개발 원칙**을 엄격히 준수하라.

## 1. Module Responsibility (역할 엄수)
- **`core/forecasting.py` (Lightweight/Fast)**:
  - 대시보드용 **초고속 예측** 모듈이다.
  - `statsmodels` 외의 무거운 ML 라이브러리 import를 금지한다.
  - 10초 이내 연산이 목표다.
- **`core/engine/trainer.py` (Compute-Intensive/Precise)**:
  - 시뮬레이션용 **정밀 예측** 모듈이다.
  - `Optuna`, `CatBoost`, `Torch` 사용을 허용한다.
  - 실행 시간이 걸리더라도 정확도가 우선이다.

## 2. Statistical Integrity (통계적 무결성)
- **Anti-Leakage**: 모델 학습(`fit`) 시, **'진행 중인 당월 데이터'**는 절대 학습셋(`train`)에 포함하지 않는다. 오직 평가(`eval`)나 진행률 비교용으로만 쓴다.
- **Business Days**: 모든 일평균(Daily Rate) 계산은 `30일`이 아닌 **`영업일(Business Day)`** 기준으로 수행하여 주말 효과를 보정한다.

## 3. UI/UX Standard
- **Error Handling**: 예측 실패 시, 화면 전체가 죽지(Crash) 않도록 `try-except` 블록으로 감싸고 '데이터 부족' 등의 대체 메시지를 보여준다.
- **On-Demand**: `Track B` 시뮬레이션은 사용자가 '실행' 버튼을 누르기 전까지 절대 자동 실행하지 않는다.