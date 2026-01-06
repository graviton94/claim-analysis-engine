# 🏛 Project Master Blueprint

## 1. Data Pipeline (Robust Extraction)

Raw 데이터의 컬럼 개수나 순서는 중요하지 않다. **오직 지정된 54개 필드**가 존재하는지만 확인하고 추출한다.

### 1.1 Extraction Logic

* **Input**: `User Upload File` (Columns: N개)
* **Logic**:
```python
target_cols = [54개_필드_리스트]
# Input에 존재하면 가져오고, 없으면 NaN 처리 (에러 발생 금지)
df_extracted = df_input.reindex(columns=target_cols)

```


* **Target Fields (54개)**:
> 접수년, 접수월, 접수일, 사업부문, 상담번호, 제품명, 제목, 분석결과, 등급기준, 불만원인,
> 대분류, 중분류, 소분류, 유통기한, 유통기한-년, 유통기한-월, 유통기한-일, 제조일자,
> 제조-년, 제조-월, 제조-일, 구입일자, 플랜트, 구입경로, 구입처, 제품군, 제품범주1,
> 제품범주2, 제품범주3, 제품코드, 개선부서명, 조치방법, 방문일자, 주소1, 성별, 연령,
> 총처리액, 보상액, 택배비용, 보상액(자소), 기타비용, 접수경로, 요구사항, LOT,
> 이물신고대상, 신고일자, 행정처분, 발생일자, 인체피해, 중대보고공유, 신속공유,
> 이물신고체크, 제품구분1, 제품구분2


* **Output**: `data/hub/` (Parquet, UTF-8-SIG)

### 1.2 Global Encoding Rule

* **읽기**: `utf-8-sig` 시도 → 실패 시 `cp949` 시도 → 실패 시 `euc-kr`
* **쓰기**: 무조건 `utf-8-sig` (Excel 호환성 및 한글 깨짐 방지)

## 2. Prediction Engine (The Champion)

단순 시계열, 통계적 예측만 사용하는 것이 아닌, **ML & DL & Optuna**가 결합된 하이브리드 경쟁 시스템.

### 2.1 Candidate Models

1. **Time-Series Stats**: `SARIMAX` (계절성/추세 반영)
2. **Machine Learning**: `CatBoost` (범주형 데이터 및 시계열 피처 강점)
3. **Deep Learning**: `LSTM` (Long Short-Term Memory, 복잡한 패턴 학습)

### 2.2 Champion Selection Flow

1. **Data Split**: Train / Validation / Test (최근 3개월)
2. **Optuna Tuning**: 각 모델별 하이퍼파라미터(Learning rate, Depth, Hidden layer 등) 50회 탐색.
3. **Evaluation**: Test 셋 기준 `RMSE` (평균 제곱근 오차) 비교.
4. **Selection**: 가장 오차가 적은 모델을 **'Current Champion'**으로 선정하여 예측 수행.

## 3. UI/UX Strategy

### 3.1 Page Flow

1. **업로드**
* 파일 던지면 → 54개 필드만 남기고 저장 완료 메시지.


2. **플랜트 분석 (Dashboard)**
* **Filter 1순위**: `플랜트` 선택 (필수)
* **Filter 2순위**: `제품군`, `불만원인` 등
* **View**: 선택된 플랜트의 월별 추이, 점유율, 워드클라우드.


3. **예측 시뮬레이션**
* "모델 학습 시작" 버튼 → Optuna 튜닝 진행률 표시 (`st.progress`)
* 최종 챔피언 모델이 예측한 **향후 6개월 그래프** + 신뢰구간 표시.
