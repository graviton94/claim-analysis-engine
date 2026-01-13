# **🏰 Risk Prediction Model \- Project Master**

## **1\. Project Overview**

* **Goal**: 클레임 데이터 기반의 실시간 리스크 모니터링(Diagnosis) 및 미래 물량 예측(Prognosis) 시스템 구축.  
* **Tech Stack**: Python 3.10+, Streamlit, Plotly, Pandas, Pyarrow, Catboost/Prophet/SARIMAX.  
* **Architecture**:  
  * **Core**: storage(I/O), analytics(Risk Logic), forecasting(Engine), etl(Validation).  
  * **Pages**: 1\_Upload, 2\_Sales, 3\_Analysis, 4\_Simulation.  
  * **Main**: app.py (Control Tower).

## **2\. Current Status (Phase 2.8 Done)**

* **Data Pipeline**: core.storage를 통한 중앙 집중식 데이터 로드 및 필터링 체계 확립.  
* **Risk Engine**: Nelson Rules \+ Zero-Filling 기법을 적용하여 0건 \-\> 급증 패턴 정밀 탐지.  
* **Dashboard**: Action-Oriented UI 및 Color System 적용으로 시인성/사용성 극대화.  
* **Consistency**: 분석(Page 3)과 예측(Page 4), 대시보드(App) 간의 데이터 및 로직 100% 일치.

## **3\. Directory Structure**

claim-analysis-engine/  
├── app.py                  \# Main Dashboard (KPI, Risk Radar, Trend)  
├── core/  
│   ├── analytics.py        \# Risk Scoring & Zero-filling Logic  
│   ├── storage.py          \# Parquet I/O & Unified Loader  
│   ├── forecasting.py      \# Forecasting Engine Wrapper  
│   └── engine/             \# ML/DL Training Modules  
├── pages/  
│   ├── 3\_플랜트\_분석.py     \# Deep Dive Diagnosis  
│   └── 4\_예측\_시뮬레이션.py  \# Future Simulation & Allocation  
├── docs/                   \# Project Documents (Source of Truth)  
└── data/hub/               \# Parquet Partitioned Data Store

## **4\. Key Checkpoints**

1. **Data Sync**: 모든 페이지는 load\_and\_filter\_data를 사용해야 함.  
2. **Risk Sync**: 리스크 점수 계산 전 반드시 prepare\_risk\_data를 통과해야 함.  
3. **UI Standard**: 새로운 기능 추가 시 정의된 Color System(\#EF151E 등)을 준수해야 함.
