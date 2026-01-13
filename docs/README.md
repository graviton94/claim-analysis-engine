# 🏆 Intelligent Claim Analysis & Early Warning System (v3.0)

> **⚠️ 작업 브랜치**: 모든 작업은 반드시 **[main 브랜치](https://github.com/graviton94/claim-analysis-engine/tree/main)**에서 수행합니다.

## 🎯 Project Vision
이 시스템은 단순 조회를 넘어, **실시간 운영(Operational)**과 **심층 분석(Analytical)**이 결합된 하이브리드 인텔리전스 플랫폼입니다.
현장 관리자에게는 즉각적인 위험 신호를, 데이터 분석가에게는 정밀한 미래 시뮬레이션 도구를 제공합니다.

## 🚀 Key Features (v3.0 Two-Track Strategy)
### 1. ⚡ Operational Engine (Track A)
- **Target**: `app.py` (대시보드), `2_통합_요약.py`
- **Tech**: `core/forecasting.py` (Ensemble of Run-rate + MoM Pattern + Light ETS)
- **Goal**: **0.5초 이내** 응답. "이번 달 마감 예상치는?"에 대한 즉답 제공.

### 2. 🧪 Simulation Lab (Track B)
- **Target**: `4_예측_시뮬레이션.py`
- **Tech**: `core/engine/trainer.py` (Optuna AutoML + CatBoost/LSTM)
- **Goal**: **정밀 타격**. "특정 제품군의 향후 6개월 시나리오는?"에 대한 딥러닝 예측 수행.

## 🖥️ Page Navigation
1. **데이터 업로드**: CSV/Excel 표준화 적재 및 연/월 파티셔닝 (Parquet).
2. **통합 요약 (Exec. View)**: 전사 리스크 스코어링 현황 및 `Quick Forecast` 요약.
3. **플랜트 정밀 분석**: 동적 피벗, **Lag(제조-접수 시차) 분석**, 이상치 하이라이팅.
4. **예측 시뮬레이션 (Lab)**: 특정 제품군(Series) 대상 **AutoML 튜닝** 및 **가상 시나리오 검증**.

## 📂 Directory Structure
```bash
claim-prediction-system/
├── core/
│   ├── analytics.py      # [Risk] 리스크 스코어링 & 이상 탐지 (Nelson Rules)
│   ├── forecasting.py    # [Fast] 대시보드용 고속 예측 엔진 (Statsmodels)
│   ├── engine/           # [Heavy] 시뮬레이션용 ML 엔진 (Optuna, CatBoost)
│   └── storage.py        # [IO] 데이터 파티셔닝 입출력
├── pages/
│   ├── 2_통합_요약.py     # Uses forecasting.py
│   └── 4_예측_시뮬레이션.py # Uses core.engine
└── app.py                # 메인 대시보드
```

## 📂 Field Definitions (54 Fields)
> **Target Fields**:
> 접수년, 접수월, 접수일, 사업부문, 상담번호, 제품명, 제목, 분석결과, 등급기준, 불만원인, 
> 대분류, 중분류, 소분류, 유통기한, 유통기한-년, 유통기한-월, 유통기한-일, 제조일자, 
> 제조-년, 제조-월, 제조-일, 구입일자, 플랜트, 구입경로, 구입처, 제품군, 제품범주1, 
> 제품범주2, 제품범주3, 제품코드, 개선부서명, 조치방법, 방문일자, 주소1, 성별, 연령, 
> 총처리액, 보상액, 택배비용, 보상액(자소), 기타비용, 접수경로, 요구사항, LOT, 
> 이물신고대상, 신고일자, 행정처분, 발생일자, 인체피해, 중대보고공유, 신속공유, 
> 이물신고체크, 제품구분1, 제품구분2
