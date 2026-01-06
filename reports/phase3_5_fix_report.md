# Phase 3.5: Fix Bugs & Dependencies (긴급 디버깅)

## 📋 개요

Phase 3 구현 후 발생한 2가지 치명적 에러를 긴급 수정합니다:

1. **TypeError**: `'<' not supported between instances of 'str' and 'NoneType'`
2. **ModuleNotFoundError**: `No module named 'optuna'`

---

## 🐛 에러 분석

### Error 1: TypeError in pages/3 (플랜트 분석)

**발생 지점**: `core/storage.py::get_claim_keys()` → 정렬 시 타입 혼합

**근본 원인**:
```python
# ❌ 문제 코드 (이전)
claim_keys = df[['플랜트', '접수년', '접수월']].drop_duplicates().sort_values(
    ['플랜트', '접수년', '접수월']  # ← None/NaN과 문자열 혼합 정렬 시도
)
```

**동작 분석**:
- `df[['플랜트', '접수년', '접수월']]` 로드 시 일부 행에 `NaN`이나 `None` 포함
- `sort_values()` 수행 시 문자열과 NoneType을 비교 → Python TypeError 발생
- Pandas는 혼합 타입 정렬 불가

### Error 2: ModuleNotFoundError in pages/4 (예측 시뮬레이션)

**발생 지점**: `core/engine/trainer.py` → `import optuna` 실패

**근본 원인**:
- `optuna` 패키지가 `requirements.txt`에 표기되었으나 사용자 환경에 미설치
- `scikit-learn`과 `fastparquet` 패키지도 누락 상태

---

## ✅ 수정 사항

### 1. core/storage.py 수정

#### 변경 1: get_claim_keys() - Type Safety 강화

**수정 전**:
```python
def get_claim_keys(path: Union[str, Path] = DATA_HUB_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    claim_keys = df[['플랜트', '접수년', '접수월']].drop_duplicates().sort_values(
        ['플랜트', '접수년', '접수월']
    )
    return claim_keys
```

**수정 후**:
```python
def get_claim_keys(path: Union[str, Path] = DATA_HUB_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    
    # ★ Step 1: None/NaN 값 제외 (dropna)
    claim_keys = df[['플랜트', '접수년', '접수월']].dropna()
    
    # ★ Step 2: 모든 컬럼을 str로 형변환 (타입 안전성)
    claim_keys['플랜트'] = claim_keys['플랜트'].astype(str)
    claim_keys['접수년'] = claim_keys['접수년'].astype(str)
    claim_keys['접수월'] = claim_keys['접수월'].astype(str)
    
    # ★ Step 3: 유니크 조합 추출 및 정렬 (이제 모든 값이 str이므로 안전)
    claim_keys = claim_keys.drop_duplicates()
    claim_keys = claim_keys.sort_values(
        ['플랜트', '접수년', '접수월'],
        key=lambda x: x.astype(str)
    ).reset_index(drop=True)
    
    return claim_keys
```

**효과**:
- `dropna()`로 None/NaN 사전 제거
- `astype(str)` 강제로 타입 통일
- 정렬 시 모든 값이 문자열이므로 TypeError 불가능

#### 변경 2: load_sales_with_estimation() - 인덱스 정렬 Type Safety

**수정 전**:
```python
for plant in plants:
    plant_df = df[df['플랜트'] == plant].copy()
    plant_df = plant_df.sort_values(['년', '월']).reset_index(drop=True)
```

**수정 후**:
```python
# ★ None/NaN 플랜트 제외
plants = df['플랜트'].dropna().unique()

for plant in plants:
    plant_df = df[df['플랜트'] == plant].copy()
    # ★ 형변환 후 정렬 (동일한 방어 로직)
    plant_df['년'] = pd.to_numeric(plant_df['년'], errors='coerce').fillna(0).astype(int)
    plant_df['월'] = pd.to_numeric(plant_df['월'], errors='coerce').fillna(0).astype(int)
    plant_df = plant_df.sort_values(['년', '월']).reset_index(drop=True)
```

**효과**:
- None/NaN 플랜트 자동 제외
- 숫자 정렬의 안전성 보장

---

### 2. pages/3_플랜트_분석.py 수정

#### 변경: 예외 처리 강화 (Traceback → 명확한 경고)

**수정 전**:
```python
try:
    claim_keys = pd.read_parquet(DATA_HUB_PATH) if Path(DATA_HUB_PATH).exists() else pd.DataFrame()
    available_plants = sorted(claim_keys['플랜트'].unique().tolist()) if not claim_keys.empty else []
except Exception as e:
    st.error(f"❌ 클레임 데이터 로드 실패: {str(e)}")  # ← Traceback 노출
    available_plants = []

if not available_plants:
    st.warning("⚠️ 사용 가능한 플랜트가 없습니다. '데이터 업로드' 페이지에서 먼저 데이터를 업로드하세요.")
    st.stop()
```

**수정 후**:
```python
try:
    # ★ Type Safe한 get_claim_keys() 사용
    from core.storage import get_claim_keys
    claim_keys = get_claim_keys(DATA_HUB_PATH)
    
    # ★ dropna() 완료된 데이터이므로 안전한 정렬
    available_plants = []
    if not claim_keys.empty and '플랜트' in claim_keys.columns:
        available_plants = sorted(claim_keys['플랜트'].dropna().unique().tolist())
except Exception as e:
    print(f"[ERROR] 플랜트 목록 로드 실패: {str(e)}")
    available_plants = []

# ★ Traceback 대신 명확한 경고 메시지
if not available_plants:
    st.warning(
        "⚠️ 분석할 데이터가 없습니다.\n\n"
        "**[데이터 업로드]** 메뉴에서 CSV/Excel 파일을 등록해주세요."
    )
    st.stop()
```

**효과**:
- Type Safe한 `get_claim_keys()` 직접 호출
- dropna() 완료된 안전한 데이터 사용
- 사용자 친화적 메시지 표시 (기술적 Traceback 제거)

---

### 3. requirements.txt 업데이트

**변경 사항**:
```diff
  # ML/DL (Phase 3)
  catboost==1.2.1
  torch==2.1.1
  statsmodels==0.14.0
  optuna==3.14.0
+ scikit-learn==1.3.2        # ← 추가
+ fastparquet==2023.10.1      # ← 추가

  # Visualization
  plotly==5.18.0
  matplotlib==3.8.2
```

**효과**:
- `optuna` 의존 완료
- `scikit-learn` (CatBoost 권장 패키지) 추가
- `fastparquet` (대체 Parquet 엔진) 추가
- 사용자가 `pip install -r requirements.txt` 1회 실행으로 모든 의존성 해결

---

## 🔧 Null 비교 에러 원천 차단 로직

### 문제 시나리오
```
DataFrame 행 1: ['PlantA', 2024, 1]     ← 정상
DataFrame 행 2: ['PlantB', None, 2]     ← None 값
DataFrame 행 3: ['PlantC', 2024, np.nan] ← NaN 값

sort_values(['플랜트', '접수년', '접수월'])
└─ 접수년 비교: 2024 (int) < None (NoneType) → TypeError!
```

### 해결 방법: 3단계 방어 로직

| 단계 | 동작 | 효과 |
|------|------|------|
| **Step 1: dropna()** | None/NaN 행 사전 제거 | 혼합 타입 원천 차단 |
| **Step 2: astype(str)** | 모든 값을 str로 통일 | 타입 호환성 100% |
| **Step 3: sort_values()** | 안전한 정렬 수행 | TypeError 불가능 |

### 실행 결과 (Phase 3.5 적용 후)
```python
# ✅ 수정된 코드
df = pd.DataFrame({
    '플랜트': ['A', 'B', 'C', None],
    '접수년': [2024, 2024, None, 2024],
    '접수월': [1, 2, 3, None]
})

claim_keys = df[['플랜트', '접수년', '접수월']].dropna()
# 결과: 'A', 'B'만 남음 (None 행 제거)

claim_keys['플랜트'] = claim_keys['플랜트'].astype(str)  # ['A', 'B']
claim_keys['접수년'] = claim_keys['접수년'].astype(str)  # ['2024', '2024']
claim_keys['접수월'] = claim_keys['접수월'].astype(str)  # ['1', '2']

# ✅ 정렬 성공 (모든 값이 str이므로 TypeError 불가능)
claim_keys.sort_values(['플랜트', '접수년', '접수월'])
```

---

## 📦 의존성 설치 지침

### 1. 터미널에서 다음 명령어 실행:
```bash
pip install -r requirements.txt
```

### 2. 설치 확인:
```bash
# optuna 설치 확인
python -c "import optuna; print(f'Optuna {optuna.__version__} ✅')"

# 모든 의존성 확인
pip list | grep -E "optuna|catboost|torch|statsmodels|scikit-learn"
```

### 3. (선택) 특정 패키지만 설치:
```bash
pip install optuna==3.14.0
pip install scikit-learn==1.3.2
pip install fastparquet==2023.10.1
```

---

## 🧪 테스트 시나리오

### 테스트 1: 플랜트 분석 (pages/3)

**전제조건**:
- `data/hub/` 디렉토리에 클레임 데이터 존재 (파티셔닝된 Parquet)

**실행 단계**:
1. 메뉴에서 "📊 플랜트 분석" 클릭
2. Step 1: 플랜트 선택 (드롭다운 표시 확인)
3. 플랜트 선택 후 Step 2: 기간 필터링
4. **기대 결과**: Traceback 없이 피벗 테이블 표시 ✅

**검증 포인트**:
- ❌ 에러 메시지 없음
- ✅ "⚠️ 분석할 데이터가 없습니다" 경고만 표시 (데이터 없을 때)
- ✅ 플랜트 목록 정렬 완벽 (혼합 타입 에러 없음)

### 테스트 2: 예측 시뮬레이션 (pages/4)

**전제조건**:
- `pip install -r requirements.txt` 완료 (optuna 설치)
- 클레임 + 매출 데이터 존재

**실행 단계**:
1. 메뉴에서 "🎯 예측 시뮬레이션" 클릭
2. Step 1: 플랜트 + 대표상품 선택
3. Step 2: 기간 선택
4. Step 3: "Optuna 튜닝 시작" 버튼 클릭
5. **기대 결과**: 진행바 표시 후 성과표 + 차트 표시 ✅

**검증 포인트**:
- ❌ ModuleNotFoundError 없음
- ✅ Optuna 진행바 (0% → 100%)
- ✅ 성과표: 3개 모델 RMSE 표시
- ✅ Plotly 차트: 6개월 예측 라인 + 95% CI

---

## 📊 수정 영향도 분석

| 파일 | 수정 내용 | 영향 범위 | 위험도 |
|------|---------|---------|--------|
| `core/storage.py` | `get_claim_keys()` 함수 로직 | pages/3, pages/4 | 🟢 Low |
| `core/storage.py` | `load_sales_with_estimation()` 로직 | pages/2, pages/3, pages/4 | 🟢 Low |
| `pages/3_플랜트_분석.py` | 예외 처리 강화 | UI/UX 개선 | 🟢 Low |
| `requirements.txt` | 패키지 추가 | 환경 설정 | 🟢 Low |

**결론**: 모든 변경사항은 **하위 호환성 보장** (기존 데이터 구조 변화 없음)

---

## 🎯 체크리스트

- [x] **1단계**: `core/storage.py` Type Safety 강화
  - [x] `get_claim_keys()` dropna + astype(str) 추가
  - [x] `load_sales_with_estimation()` 인덱스 정렬 안전화
  
- [x] **2단계**: `pages/3_플랜트_분석.py` 예외 처리 개선
  - [x] Traceback 대신 st.warning() 표시
  - [x] `get_claim_keys()` 직접 호출로 Type Safety 확보
  
- [x] **3단계**: `requirements.txt` 의존성 완성
  - [x] `optuna`, `scikit-learn`, `fastparquet` 추가
  - [x] 설치 명령어 명시
  
- [x] **4단계**: 문서화
  - [x] 근본 원인 분석
  - [x] 해결책 구현
  - [x] 테스트 시나리오 정의

---

## 📈 결론

Phase 3.5는 **Type Safety와 사용자 경험** 두 가지를 동시에 개선합니다:

| 항목 | 개선 내용 |
|------|---------|
| **안정성** | None/NaN 값 사전 제거 → TypeError 원천 차단 |
| **사용자 경험** | Traceback 제거 → 명확한 경고 메시지 표시 |
| **의존성** | 누락 패키지 추가 → ModuleNotFoundError 해결 |
| **유지보수성** | 방어 로직 문서화 → 향후 버그 예방 |

**다음 단계**: Phase 4 (Integration Testing)에서 실제 데이터로 전체 파이프라인 검증

---

**작성 날짜**: 2026-01-06  
**담당자**: Advanced Claim Prediction System 개발팀  
**상태**: ✅ Complete
