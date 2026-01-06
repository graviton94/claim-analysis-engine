# 🏆 Advanced Claim Prediction System (ML/DL/Optuna)
> **⚠️ 작업 브랜치 주의**: 모든 개발 및 커밋은 반드시 **[Main Branch](https://github.com/graviton94/claim-analysis-engine/tree/main)**에서 수행해야 합니다.
> **"Global ML/DL 챔피언 모델 기반, 플랜트 중심 정밀 분석 시스템"**
> 단순 통계나 EWS를 넘어, Optuna로 튜닝된 최적의 모델(CatBoost, LSTM, SARIMAX)을 경쟁시켜 미래 리스크를 예측합니다.

## 🎯 Project Goal
1. **Hyper-Model**: 머신러닝(ML)과 딥러닝(DL) 모델을 Optuna로 자동 튜닝하고, 성능이 가장 좋은 'Champion Model'을 선정하여 예측.
2. **Plant-Centric & Pivot**: **'플랜트'**를 최상위 기준으로 하되, 사용자가 엑셀 피벗테이블처럼 **보고 싶은 열(Column)을 자유롭게 구성**하는 유연한 대시보드.
3. **Sales Management**: 단순 건수 분석을 넘어, **매출 수량 대비 클레임율(PPM)**을 분석하기 위한 매출 데이터 통합 관리.
4. **Robust ETL**: Input 데이터가 100개든 200개든 상관없이, **핵심 54개 필드**만 추출하여 연/월 단위로 파티셔닝 적재.

## 🏗 Tech Stack
- **Core**: Python 3.10+
- **ML/DL**: `CatBoost`, `PyTorch` (LSTM), `Statsmodels` (SARIMAX), `Optuna` (AutoML)
- **Data**: Pandas, Pyarrow (Parquet Partitioning)
- **UI**: Streamlit (Data Editor, Pivot Table UI)
- **Viz**: Plotly Express / Graph Objects

## 📂 Directory Structure
```bash
claim-prediction-system/
├── app.py                  # 메인 엔트리
├── core/
│   ├── config.py           # 54개 필드 정의
│   ├── etl.py              # 필드 추출 및 1행=1건 처리
│   ├── storage.py          # Parquet 파티셔닝 입출력
│   └── engine/             # 예측 엔진 패키지
├── pages/
│   ├── 1_데이터_업로드.py    # 클레임 데이터 적재
│   ├── 2_매출수량_관리.py    # [NEW] 플랜트별 매출 입력/수정
│   ├── 3_플랜트_분석.py      # [UPDATE] 피벗 스타일 대시보드
│   └── 4_예측_시뮬레이션.py  # 챔피언 모델 학습 및 예측
├── data/
│   ├── hub/                # 클레임 데이터 (접수년/월 파티셔닝)
│   ├── sales/              # [NEW] 매출 데이터 저장소
│   └── models/             # 학습된 모델 저장소
└── docs/                   # 설계 문서

## 📂 Field def
> **Target Fields**:
> 접수년, 접수월, 접수일, 사업부문, 상담번호, 제품명, 제목, 분석결과, 등급기준, 불만원인, 
> 대분류, 중분류, 소분류, 유통기한, 유통기한-년, 유통기한-월, 유통기한-일, 제조일자, 
> 제조-년, 제조-월, 제조-일, 구입일자, 플랜트, 구입경로, 구입처, 제품군, 제품범주1, 
> 제품범주2, 제품범주3, 제품코드, 개선부서명, 조치방법, 방문일자, 주소1, 성별, 연령, 
> 총처리액, 보상액, 택배비용, 보상액(자소), 기타비용, 접수경로, 요구사항, LOT, 
> 이물신고대상, 신고일자, 행정처분, 발생일자, 인체피해, 중대보고공유, 신속공유, 
> 이물신고체크, 제품구분1, 제품구분2
