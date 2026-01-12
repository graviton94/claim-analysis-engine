# 🏛️ Project Master Blueprint (v3.0)

## 0. Fundamental Principles
- **Data-Driven**: 모든 판단은 데이터에 근거하며, 감(Gut feeling)을 통계적 수치로 변환한다.
- **Speed**: 어떤 분석이든 1초 이내 응답을 목표로 파티셔닝과 캐싱을 최적화한다.

## 1. Data Strategy: Smart Partitioning
- **Storage**: `Year/Month` 파티셔닝된 Parquet 파일로 관리.
- **Series Mart**: 분석 속도를 위해 `[플랜트|대분류|소분류]` 단위의 Nested JSON Series 별도 생성.
- **Schema**: 54개 표준 필드 준수 (ETL 과정에서 타입 강제 변환).

## 2. Core Engines (The Brain)

### 2.1 🛡️ Risk Scoring Engine (`analytics.py`)
과거 데이터 분포와 현재 추세를 비교하여 **'위험도(Risk Score)'**를 산출한다.
- **Statistical Guards**: 소량 데이터의 과대 해석 방지 (Small Sample Guard).
- **Nelson Rules**: 공정 관리도(Control Chart) 기법을 응용한 8가지 이상 패턴 감지.
- **Score Logic**: `기본 점수` + `패턴 가중치` + `확률적 임계치 초과 보너스`.

### 2.2 🔭 Forecasting Engine (`forecasting.py`)
미래 물량을 예측하고, 현재의 진행 속도가 적절한지 판단한다.
- **Input Guard**: 마감되지 않은 당월 데이터를 학습셋에서 자동 제외 (`Training Set Isolation`).
- **Biz-Day Logic**: `np.busday_count`를 활용한 정밀한 일평균(Run-rate) 계산.
- **Adaptive Weight**: 월초에는 `과거 패턴(MoM)` 중심, 월말에는 `현재 실적(Run-rate)` 중심으로 가중치 동적 조절.
- **Model Pool**: Holt-Winters (Trend+Seasonality) 및 ARIMA 자동 선택.

## 3. Advanced UX/UI
- **Dynamic Pivot**: 사용자가 행/열을 자유롭게 드래그 앤 드롭하듯 변경 가능 (`3_플랜트_분석`).
- **Explainable AI (XAI)**: 단순히 "위험함"이 아니라, "왜 위험한지(Why)"를 텍스트로 풀어서 제공 (`format_diagnosis`).