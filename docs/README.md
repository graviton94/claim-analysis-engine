# **🏆 Intelligent Claim Analysis & Early Warning System (v3.5)**

**⚠️ 작업 브랜치**: claim-analysis-engine-simul\_refactor (Phase 2.8 Complete)

## **🎯 Project Vision**

이 시스템은 단순 조회를 넘어, \*\*실시간 운영(Operational)\*\*과 \*\*심층 분석(Analytical)\*\*이 결합된 하이브리드 인텔리전스 플랫폼입니다.  
현장 관리자에게는 즉각적인 위험 신호와 Action Item을, 데이터 분석가에게는 정밀한 미래 시뮬레이션 도구를 제공합니다.

## **🚀 Key Features (v3.5 Two-Track Strategy)**

### **1\. ⚡ Operational Engine (Dashboard & Alert)**

* **Target**: app.py (Quality Control Tower)  
* **Tech**: core/analytics.py (Nelson Rules \+ Zero-Filling Logic)  
* **Features**:  
  * **Risk Radar**: 실시간 위험(🔴)/주의(🟡) 등급 자동 분류 및 시각화.  
  * **Action Card**: 카드 내에서 즉시 엑셀 다운로드 및 정밀 분석 페이지로 이동.  
  * **Zero-Friction Linking**: 별도 설정 없이 클릭 한 번으로 필터가 적용된 상세 분석 화면 진입.

### **2\. 🧪 Simulation Lab (Deep Dive)**

* **Target**: 4\_예측\_시뮬레이션.py, 3\_플랜트\_분석.py  
* **Tech**: core/engine/trainer.py (Ensemble: Prophet \+ AutoML \+ SARIMAX)  
* **Features**:  
  * **Diagnosis (Page 3\)**: 동적 피벗 및 Lag(제조-접수 시차) 분석.  
  * **Prognosis (Page 4\)**: 모델 경합(Competition)을 통한 최적 예측 모델 선정 및 소분류 배분(Allocation).

## **🖥️ Page Navigation**

1. **Quality Control Tower (Main)**: 전사 KPI, Trend 차트, Risk Radar, Critical LOT 체크.  
2. **데이터 업로드**: CSV/Excel 표준화 적재 및 연/월 파티셔닝 (Parquet).  
3. **매출수량 관리**: 제품 판매량 데이터 CRUD (클레임률 산출용).  
4. **플랜트 정밀 분석**: 특정 이슈에 대한 다차원 드릴다운 및 원인 규명.  
5. **예측 시뮬레이션**: 미래 리스크 물량 예측 및 시나리오 검증.

## **📂 Directory Structure (Core Centric)**
```bash
claim-prediction-system/  
├── core/  
│   ├── analytics.py      \# \[Risk\] 리스크 스코어링 & Zero-Filling (공통 로직)  
│   ├── forecasting.py    \# \[Fast\] 대시보드용 고속 예측 엔진  
│   ├── engine/           \# \[Heavy\] 시뮬레이션용 ML 엔진 (Optuna, CatBoost)  
│   └── storage.py        \# \[IO\] 통합 데이터 로더 (load\_and\_filter\_data)  
├── pages/  
│   ├── 3\_플랜트\_분석.py   \# Uses core.analytics & storage  
│   └── 4\_예측\_시뮬레이션.py \# Uses core.engine & analytics  
└── app.py                \# 메인 대시보드 (All Core Modules Integrated)
```
## **🎨 Design System**

* **Danger**: \#EF151E (심각한 리스크)  
* **Caution**: \#FF9700 (주의 단계)  
* **Safe/Link**: \#006ECD (정상, 바로가기)

## **📂 Field Definitions (54 Fields)**
> **Target Fields**:
> 접수년, 접수월, 접수일, 사업부문, 상담번호, 제품명, 제목, 분석결과, 등급기준, 불만원인, 
> 대분류, 중분류, 소분류, 유통기한, 유통기한-년, 유통기한-월, 유통기한-일, 제조일자, 
> 제조-년, 제조-월, 제조-일, 구입일자, 플랜트, 구입경로, 구입처, 제품군, 제품범주1, 
> 제품범주2, 제품범주3, 제품코드, 개선부서명, 조치방법, 방문일자, 주소1, 성별, 연령, 
> 총처리액, 보상액, 택배비용, 보상액(자소), 기타비용, 접수경로, 요구사항, LOT, 
> 이물신고대상, 신고일자, 행정처분, 발생일자, 인체피해, 중대보고공유, 신속공유, 
> 이물신고체크, 제품구분1, 제품구분2
