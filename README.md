# 🏆 Advanced Claim Prediction System (ML/DL/Optuna)

> **"Global ML/DL 챔피언 모델 기반, 플랜트 중심 정밀 분석 시스템"**
> 단순 통계나 EWS를 넘어, Optuna로 튜닝된 최적의 모델(CatBoost, LSTM, SARIMAX)을 경쟁시켜 미래 리스크를 예측합니다.

## 🎯 Project Goal
1. **Hyper-Model**: SARIMAX, 가중평균, 시계열 분석 및 머신러닝(ML)과 딥러닝(DL) 모델을 Optuna로 자동 튜닝하고, 성능이 가장 좋은 'Champion Model'을 선정하여 예측.
2. **Plant-Centric**: 전사 비교가 아닌, **내가 선택한 '플랜트'**의 데이터만 깊게 파고드는 심층 대시보드.
3. **Robust ETL**: Input 데이터가 100개든 200개든 상관없이, **핵심 54개 필드**만 추출하여 적재.
4. **Korean Native**: 모든 파이프라인에서 **한글 인코딩(`utf-8-sig`)** 완벽 보장.

## 🏗 Tech Stack
- **Core**: Python 3.10+
- **ML/DL**: `CatBoost`, `PyTorch` (LSTM), `Statsmodels` (SARIMAX), `Optuna` (AutoML)
- **Data**: Pandas, Pyarrow (Parquet)
- **UI**: Streamlit
- **Viz**: Plotly Express / Graph Objects

## 📂 Directory Structure
```bash
claim-prediction-system/
├── app.py                  # 메인 엔트리
├── core/
│   ├── config.py           # 54개 필드 정의
│   ├── etl.py              # [핵심] 필드 추출 및 인코딩 처리
│   ├── storage.py          # Parquet 입출력
│   └── engine/             # 예측 엔진 패키지
│       ├── trainer.py      # 모델 학습 및 Optuna 튜닝
│       ├── models.py       # ML/DL 모델 정의 (CatBoost, LSTM 등)
│       └── selector.py     # 챔피언 모델 선정 로직
├── pages/
│   ├── 1_데이터_업로드.py    # 54개 필드 강제 추출
│   ├── 2_플랜트_분석.py      # 플랜트 중심 현황 분석
│   └── 3_예측_시뮬레이션.py  # 학습 및 예측 결과 뷰
├── data/
│   ├── hub/                # 통합 데이터 (Parquet)
│   └── models/             # 학습된 모델(.pkl, .pth) 저장소
└── docs/                   # 설계 문서
