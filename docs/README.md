# 🏆 Intelligent Claim Analysis & Early Warning System (v3.0)

> **⚠️ 작업 브랜치**: 모든 작업은 반드시 **[main 브랜치](https://github.com/graviton94/claim-analysis-engine/tree/main)**에서 수행합니다.

## 🎯 Project Vision
이 시스템은 단순한 대시보드가 아닙니다. **비즈니스 규칙(Rule)**, **통계적 이상 탐지(Statistical Detection)**, 그리고 **시계열 머신러닝(Time-series ML)**이 결합된 **하이브리드 인텔리전스 플랫폼**입니다. 
수만 개의 제품 시리즈를 24시간 감시하며, "왜 위험한지"에 대한 통계적 근거와 "앞으로 어떻게 될지"에 대한 예측 시나리오를 제공합니다.

## 🚀 Key Features (v3.0)
### 1. 🚨 Dual-Track Risk Scoring (`core/analytics.py`)
- **Nelson Rules**: 연속 상승, 편향 등 이상 패턴 감지.
- **Distribution Analysis**: 포아송/음이항 분포 기반의 희소 사건(Rare Event) 확률 계산.
- **Anomaly Detection**: IQR 및 STL 분해를 통한 시계열 이상치 자동 발굴.

### 2. 🔮 Advanced Forecasting Engine (`core/forecasting.py`)
- **Ensemble Prediction**: `Run-rate`(현재 실적) + `MoM Pattern`(과거 계절성) + `Holt-Winters/ARIMA`(시계열 모델) 앙상블.
- **Business Day Awareness**: 영업일 기준 진행률 계산으로 월초/공휴일 예측 왜곡 방지.
- **Data Guard**: 학습 데이터와 평가 데이터의 엄격한 분리(Data Leakage 방지).

## 🖥️ Page Navigation
1. **데이터 업로드**: CSV/Excel 표준화 적재 및 연/월 파티셔닝 저장 (Parquet).
2. **통합 요약 (Executive Summary)**: 전사 리스크 스코어링 현황 및 고위험군 Top-N 리포트.
3. **플랜트 정밀 분석**: 동적 피벗 테이블, **Lag(제조-접수 시차) 분석**, 이상치 하이라이팅.
4. **예측 시뮬레이션**: Optuna 기반 챔피언 모델(CatBoost/LSTM/SARIMAX) 경합 및 3개월 단기 예보.

## 📂 Directory Structure
```bash
claim-prediction-system/
├── core/
│   ├── analytics.py      # [Engine 1] 리스크 스코어링 & 이상 탐지 (Nelson, STL)
│   ├── forecasting.py    # [Engine 2] 시계열 예측 앙상블 (ETS, ARIMA, Ensemble)
│   ├── etl.py            # 데이터 전처리 및 표준화
│   └── storage.py        # 고성능 파티셔닝 입출력 (Arrow/Parquet)
├── pages/
│   ├── 1_데이터_업로드.py
│   ├── 2_통합_요약.py     
│   ├── 3_플랜트_분석.py   # Lag 분석 & 동적 피벗 탑재
│   └── 4_예측_시뮬레이션.py # Optuna 튜닝 및 시뮬레이션
└── app.py                # 메인 대시보드 (Risk & Forecast 요약)
```

## 📂 Field def
> **Target Fields**:
> 접수년, 접수월, 접수일, 사업부문, 상담번호, 제품명, 제목, 분석결과, 등급기준, 불만원인, 
> 대분류, 중분류, 소분류, 유통기한, 유통기한-년, 유통기한-월, 유통기한-일, 제조일자, 
> 제조-년, 제조-월, 제조-일, 구입일자, 플랜트, 구입경로, 구입처, 제품군, 제품범주1, 
> 제품범주2, 제품범주3, 제품코드, 개선부서명, 조치방법, 방문일자, 주소1, 성별, 연령, 
> 총처리액, 보상액, 택배비용, 보상액(자소), 기타비용, 접수경로, 요구사항, LOT, 
> 이물신고대상, 신고일자, 행정처분, 발생일자, 인체피해, 중대보고공유, 신속공유, 
> 이물신고체크, 제품구분1, 제품구분2
