# 🏛️ Project Master Blueprint (v3.0)

## 0. Fundamental Principles
- **Separation of Concerns**: 운영용 조회와 분석용 연산은 철저히 분리한다. 무거운 연산이 대시보드 속도를 저하시켜선 안 된다.

## 1. Data Strategy
- **파티셔닝 규칙**: 모든 허브 데이터는 `data/hub/YYYY/MM/data.parquet` 구조로 연·월 단위 파티셔닝한다.
- **조회 원칙**: 연·월 필터 없이 전체 데이터셋을 한 번에 적재하지 않는다. 항상 `년(접수년)`, `월(접수월)` 조건을 기준으로 범위를 한정한다.
- **I/O 유틸**: 기간 조회 시에는 반드시 `core/storage.load_range(start_ym, end_ym)`를 사용하여 효율적인 범위 로딩을 보장한다.

## 2. Prediction Architecture: Two-Track Strategy

### 2.1 Track A: Operational Forecasting (운영용)
- **Role**: 현황판, 조기 경보.
- **Engine**: `core/forecasting.py`
- **Logic**: 
  - **Ensemble**: `Run-rate`(실적) + `Pattern`(계절성) + `ETS`(추세) 가중 평균.
  - **Dynamic Weight**: 월초에는 과거 패턴 중시, 월말에는 실적 중시.
- **Constraint**: 무거운 라이브러리(Torch, Optuna) 사용 금지. 오직 `numpy`, `statsmodels`만 허용.

### 2.2 Track B: Strategic Simulation (분석용)
- **Role**: 심층 원인 분석, 미래 설계.
- **Engine**: `core/engine/trainer.py`
- **Logic**:
  - **AutoML**: Optuna를 통해 Hyperparameter(Trend 강도, 계절 주기 등) 자동 최적화.
  - **Competition**: Prophet vs CatBoost vs LSTM 성능 경합 후 챔피언 모델 선정.
- **UI UX**: 사용자가 버튼을 눌렀을 때만(On-demand) 연산 시작.

## 3. Analysis Intelligence
- **Risk Scoring**: 단순 건수가 아닌, 포아송 분포 기반의 **희소 사건 확률**과 Nelson Rules를 결합하여 산출 (`core/analytics.py`).
- **Explainability**: "위험함"이 아니라 "최근 3개월 연속 상승하여 위험함"이라는 구체적 사유 텍스트 생성.