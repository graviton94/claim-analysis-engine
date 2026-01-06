# 🏛 Project Master Blueprint

## 0. Git Branch Policy
- **Target**: `https://github.com/graviton94/claim-analysis-engine/tree/main`
- **Rule**: 모든 코드 변경사항은 `main` 브랜치에 직접 커밋하거나, `main`으로 향하는 PR이어야 한다.

## 1. Data Pipeline Strategy

### 1.1 Robust Extraction & Partitioning
- **Input**: User Upload File (Columns: N개)
- **Logic**:
  1. `core/config.py`의 **54개 필드**만 추출 (없는 필드는 Null 처리).
  2. **1행 = 1클레임 건** 원칙 준수 (임의 집계 금지).
  3. `접수년`, `접수월` 컬럼을 기준으로 파티셔닝 저장.
- **Output**: `data/hub/접수년=YYYY/접수월=MM/part-0.parquet`
  - *이점: 대용량 데이터도 연/월 단위로 쪼개져 있어 조회 속도 빠름, 예측 모델 학습 시 특정 기간만 로딩 가능.*
> **Target Fields**:
> 접수년, 접수월, 접수일, 사업부문, 상담번호, 제품명, 제목, 분석결과, 등급기준, 불만원인, 
> 대분류, 중분류, 소분류, 유통기한, 유통기한-년, 유통기한-월, 유통기한-일, 제조일자, 
> 제조-년, 제조-월, 제조-일, 구입일자, 플랜트, 구입경로, 구입처, 제품군, 제품범주1, 
> 제품범주2, 제품범주3, 제품코드, 개선부서명, 조치방법, 방문일자, 주소1, 성별, 연령, 
> 총처리액, 보상액, 택배비용, 보상액(자소), 기타비용, 접수경로, 요구사항, LOT, 
> 이물신고대상, 신고일자, 행정처분, 발생일자, 인체피해, 중대보고공유, 신속공유, 
> 이물신고체크, 제품구분1, 제품구분2

### 1.2 Sales Data Management (New)
- **Input**: Streamlit `data_editor`를 통한 사용자 직접 입력/수정.
- **Schema**: `[플랜트, 년, 월, 매출수량]`
- **Storage**: `data/sales/sales_history.parquet` (단일 파일 또는 연단위 분할)
- **Join**: 분석 시 `플랜트+년+월`을 Key로 클레임 데이터와 결합하여 **PPM(Parts Per Million)** 산출.

## 2. UI/UX Strategy: Pivot & Plant-Centric

### 2.1 Page 3: 플랜트 분석 (Dashboard)
**"사용자가 만드는 피벗 테이블"**
1. **Top Filter (필수)**: `플랜트` 선택 (단일/다중).
2. **Period Filter**: 조회할 `접수년월` 범위 (Start ~ End).
3. **Pivot UI**:
   - **행(Index)**: `접수년`, `접수월` (기본 고정)
   - **열(Columns)**: 사용자 멀티 선택 (예: `제품군`, `불만원인`, `대분류` 등)
   - **값(Values)**: `건수` (Count), `매출대비율` (Calc), `보상액` (Sum)
4. **View**: 구성된 피벗 테이블 표출 및 시계열 차트 연동.

### 2.2 Page 2: 매출수량 관리
- **기능**: 엑셀처럼 생긴 그리드(`st.data_editor`) 제공.
- **Action**: 행 추가/삭제/수정 후 '저장' 버튼 클릭 시 Parquet 갱신.

## 3. Prediction Engine (The Champion)
- **Data Loading**: 사용자가 선택한 파티션(기간)의 데이터를 로드.
- **Feature Engineering**: `매출수량`이 있다면 이를 피처로 활용(1000개당 클레임 수 등)하여 예측 정확도 향상.
- **Process**: Train/Test Split → Optuna Tuning → RMSE Comparison → Champion Selection.
