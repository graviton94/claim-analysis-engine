# Phase 2: Deep Dive Analysis 상세 개발 계획서

**작성일**: 2026-01-06  
**대상 Phase**: Phase 2 - Deep Dive Analysis (P3 플랜트 분석 고도화)  
**작성자**: Advanced Claim Analysis & Early Warning System v2.0 개발팀  
**Branch**: `main`

---

## 📋 목차

1. [개요](#1-개요)
2. [P3 UI 구조 재설계](#2-p3-ui-구조-재설계)
3. [이상치(Outlier) 감지 로직](#3-이상치outlier-감지-로직)
4. [Lag 분석 구현](#4-lag-분석-구현)
5. [Data Mart 연동](#5-data-mart-연동)
6. [예측 엔진과의 접점](#6-예측-엔진과의-접점)
7. [개발 일정](#7-개발-일정)
8. [기술 스택](#8-기술-스택)

---

## 1. 개요

### 1.1 배경

현재 `pages/3_플랜트_분석.py`는 6-Step Adaptive Dashboard 구조로 구현되어 있으나, HQ가 지시한 고도화 요구사항을 반영하기 위해 다음 기능들의 추가 개발이 필요합니다:

- **이상치 감지**: 피벗 테이블에서 통계적으로 비정상적인 값 자동 탐지 및 시각적 강조
- **Lag 분석**: 제조일자와 접수일자의 시차 분석 및 시각화
- **Data Mart 최적화**: `data/series/` 폴더 기반 분절 데이터 효율적 로드
- **예측 연동**: P3 분석 결과를 P4 예측 엔진과 연계

### 1.2 목표

Phase 2 개발을 통해 다음을 달성합니다:

1. ✅ **사용자 경험 향상**: 이상치를 자동으로 탐지하여 분석가의 의사결정 지원
2. ✅ **분석 깊이 강화**: Lag 분석을 통한 프로세스 지연 요인 파악
3. ✅ **성능 최적화**: Data Mart 전략으로 대용량 데이터 처리 속도 개선
4. ✅ **시스템 통합**: P3 ↔ P4 간 데이터 흐름 표준화

---

## 2. P3 UI 구조 재설계

### 2.1 현재 구조 (As-Is)

`pages/3_플랜트_분석.py`는 현재 다음과 같은 6단계 흐름을 구현하고 있습니다:

```
Step 1-2: 플랜트 선택 + 데이터 요약
   ↓
Step 3: 필터 설정 (4대 필터: 대분류, 사업부문, 등급기준, 불만원인)
   ↓
Step 4: 피벗 설정 (행 인덱스 컬럼 선택)
   ↓
Step 5: 지표 선택 + 매크로 버튼 ("실적만 보기")
   ↓
Step 6: 분석 시작 → 결과 시각화 (피벗 테이블 + 차트)
```

### 2.2 개선 계획 (To-Be)

#### 2.2.1 매크로 버튼 로직 강화

**현재 구현**:
- "실적만 보기" 체크박스: 사업부문(식품, B2B식품) + 불만원인(제조불만, 고객불만족, 구매불만) 자동 설정

**개선 사항**:
```python
# core/config.py에 매크로 정의 추가
PERFORMANCE_MACROS = {
    "실적만보기": {
        "사업부문": ["식품", "B2B식품"],
        "불만원인": ["제조불만", "고객불만족", "구매불만"]
    },
    "품질이슈": {
        "대분류": ["관능", "이물", "변질"],
        "등급기준": ["A", "B"]
    },
    "고객대응": {
        "불만원인": ["고객불만족", "서비스불만"],
        "등급기준": ["A"]
    }
}
```

**구현 위치**: `pages/3_플랜트_분석.py` Step 5

```python
# Step 5에 매크로 선택 드롭다운 추가
with col_check3:
    selected_macro = st.selectbox(
        "⚡ 매크로 선택",
        ["없음"] + list(PERFORMANCE_MACROS.keys()),
        key="macro_selector"
    )
    
    if selected_macro != "없음":
        macro_config = PERFORMANCE_MACROS[selected_macro]
        # 필터 자동 설정 (Step 3 필터 덮어쓰기)
        for filter_key, filter_values in macro_config.items():
            st.session_state[f"filter_{filter_key.lower()}"] = filter_values
        st.info(f"✅ 매크로 '{selected_macro}' 적용됨")
```

#### 2.2.2 Step 6 결과 섹션 확장

**기존**:
- 피벗 테이블 (건수/PPM)
- 시계열 차트

**추가 예정**:
1. **이상치 탐지 결과 테이블** (새 섹션)
2. **Lag 분석 차트** (히스토그램 + 통계량)
3. **예측 연동 버튼** ("P4에서 예측 실행")

```python
# Step 6-F: 이상치 탐지 (신규)
st.write("#### 🚨 이상치 탐지 결과")
outliers_df = detect_outliers(ppm_data, method='IQR', threshold=1.5)
if not outliers_df.empty:
    st.dataframe(outliers_df.style.applymap(
        lambda x: 'background-color: #ff6b6b' if x in outliers_df['이상치값'] else ''
    ))
else:
    st.success("✅ 이상치가 발견되지 않았습니다.")

# Step 6-G: Lag 분석 (신규)
st.write("#### ⏱️ Lag 분석 (제조일자 ~ 접수일자)")
lag_result = analyze_lag(filtered_claims)
fig_lag = create_lag_histogram(lag_result)
st.plotly_chart(fig_lag)
st.metric("평균 Lag", f"{lag_result['mean_lag']:.1f}일")
st.metric("중앙값 Lag", f"{lag_result['median_lag']:.1f}일")
```

### 2.3 설정 저장 전략

**현재**:
- JSON 파일 (`data/plant_settings.json`)에 플랜트별 Step 3, 4 설정 저장

**유지**:
- 동일한 전략 유지 (변경 없음)
- Step 5 매크로 선택은 저장하지 않음 (매번 사용자가 선택)

---

## 3. 이상치(Outlier) 감지 로직

### 3.1 알고리즘 선택

#### 3.1.1 IQR (Interquartile Range) 방식

**수식**:
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR

이상치: x < Lower Bound or x > Upper Bound
```

**장점**:
- 간단하고 직관적
- 중앙값 기반이므로 극단값에 강건
- 박스플롯과 일관성

**구현**:
```python
# core/analytics.py (신규 파일)
def detect_outliers_iqr(
    df: pd.DataFrame,
    value_col: str = '건수',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    IQR 방식으로 이상치 탐지.
    
    Args:
        df: 피벗 테이블 데이터
        value_col: 분석 대상 컬럼
        threshold: IQR 배수 (기본 1.5)
    
    Returns:
        pd.DataFrame: 이상치 정보 (행 인덱스, 값, Lower/Upper Bound)
    """
    Q1 = df[value_col].quantile(0.25)
    Q3 = df[value_col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = df[
        (df[value_col] < lower_bound) | (df[value_col] > upper_bound)
    ].copy()
    
    outliers['Lower_Bound'] = lower_bound
    outliers['Upper_Bound'] = upper_bound
    outliers['이상치_유형'] = outliers[value_col].apply(
        lambda x: '높음' if x > upper_bound else '낮음'
    )
    
    return outliers
```

#### 3.1.2 Z-Score 방식 (선택적)

**수식**:
```
Z = (x - μ) / σ
이상치: |Z| > 3 (또는 2.5)
```

**장점**:
- 정규분포 가정 시 정확
- 표준화된 척도 제공

**단점**:
- 극단값에 민감 (평균/표준편차 영향)
- 소규모 데이터셋에서 부정확

**구현**:
```python
def detect_outliers_zscore(
    df: pd.DataFrame,
    value_col: str = '건수',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Z-Score 방식으로 이상치 탐지.
    """
    mean = df[value_col].mean()
    std = df[value_col].std()
    
    df['Z_Score'] = (df[value_col] - mean) / std
    outliers = df[abs(df['Z_Score']) > threshold].copy()
    
    return outliers
```

### 3.2 UI 시각화 전략

#### 3.2.1 피벗 테이블 하이라이트

**Streamlit DataFrame Styling**:
```python
def style_outliers(df: pd.DataFrame, outlier_indices: list) -> pd.DataFrame:
    """
    이상치 행을 빨간색 배경으로 강조.
    """
    def highlight_row(row):
        if row.name in outlier_indices:
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)
    
    return df.style.apply(highlight_row, axis=1)

# 사용 예시
styled_df = style_outliers(count_pivot, outliers_df.index.tolist())
st.dataframe(styled_df, use_container_width=True)
```

#### 3.2.2 이상치 전용 테이블

**구성**:
| 대분류 | 중분류 | 소분류 | 년월 | 실제값 | Lower Bound | Upper Bound | 유형 |
|--------|--------|--------|------|--------|-------------|-------------|------|
| 관능   | 향     | 이취   | 2024-08 | 150 | 30 | 100 | 높음 |

```python
st.dataframe(
    outliers_df[[
        '대분류', '중분류', '소분류', '년월', '건수',
        'Lower_Bound', 'Upper_Bound', '이상치_유형'
    ]],
    use_container_width=True
)
```

### 3.3 알고리즘 선택 UI

**Step 5에 옵션 추가**:
```python
with st.expander("🔧 이상치 탐지 설정", expanded=False):
    outlier_method = st.radio(
        "탐지 방법",
        ["IQR", "Z-Score"],
        horizontal=True,
        help="IQR: 박스플롯 기반 (권장), Z-Score: 정규분포 가정"
    )
    
    if outlier_method == "IQR":
        iqr_threshold = st.slider("IQR 배수", 1.0, 3.0, 1.5, 0.1)
    else:
        zscore_threshold = st.slider("Z-Score 임계값", 2.0, 4.0, 3.0, 0.1)
```

---

## 4. Lag 분석 구현

### 4.1 Lag 정의

**Lag** = 접수일자 - 제조일자 (단위: 일)

**의미**:
- 양수(+): 제조 후 시간이 지나서 접수 (정상)
- 0: 제조 당일 접수
- 음수(-): 접수일이 제조일보다 이전 (데이터 오류 또는 특수 케이스)

### 4.2 계산 로직

#### 4.2.1 데이터 전처리

```python
# core/analytics.py
def calculate_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    제조일자와 접수일자 사이의 Lag 계산.
    
    Args:
        df: 클레임 데이터 (제조일, 접수일 컬럼 포함)
    
    Returns:
        pd.DataFrame: Lag 컬럼 추가된 데이터
    """
    df = df.copy()
    
    # 날짜 컬럼 변환 (문자열 → datetime)
    df['제조일_dt'] = pd.to_datetime(df['제조일'], errors='coerce')
    df['접수일_dt'] = pd.to_datetime(df['접수일'], errors='coerce')
    
    # Lag 계산 (일 단위)
    df['Lag_일수'] = (df['접수일_dt'] - df['제조일_dt']).dt.days
    
    # NaT 처리 (계산 불가능한 행)
    df = df.dropna(subset=['Lag_일수'])
    
    return df
```

#### 4.2.2 통계량 산출

```python
def analyze_lag(df: pd.DataFrame) -> dict:
    """
    Lag 분석 통계량 산출.
    
    Returns:
        dict: {mean_lag, median_lag, std_lag, min_lag, max_lag,
               negative_count, zero_count, positive_count}
    """
    lag_series = df['Lag_일수']
    
    return {
        'mean_lag': lag_series.mean(),
        'median_lag': lag_series.median(),
        'std_lag': lag_series.std(),
        'min_lag': lag_series.min(),
        'max_lag': lag_series.max(),
        'negative_count': (lag_series < 0).sum(),
        'zero_count': (lag_series == 0).sum(),
        'positive_count': (lag_series > 0).sum(),
        'total_count': len(lag_series)
    }
```

### 4.3 시각화

#### 4.3.1 히스토그램

```python
import plotly.express as px

def create_lag_histogram(df: pd.DataFrame) -> go.Figure:
    """
    Lag 분포 히스토그램 생성.
    """
    fig = px.histogram(
        df,
        x='Lag_일수',
        nbins=50,
        title='Lag 분포 (제조일 ~ 접수일)',
        labels={'Lag_일수': 'Lag (일)', 'count': '건수'},
        color_discrete_sequence=['#3498db']
    )
    
    # 평균선 추가
    mean_lag = df['Lag_일수'].mean()
    fig.add_vline(
        x=mean_lag,
        line_dash="dash",
        line_color="red",
        annotation_text=f"평균: {mean_lag:.1f}일"
    )
    
    # 중앙값선 추가
    median_lag = df['Lag_일수'].median()
    fig.add_vline(
        x=median_lag,
        line_dash="dot",
        line_color="green",
        annotation_text=f"중앙값: {median_lag:.1f}일"
    )
    
    fig.update_xaxes(title_text="Lag (일)")
    fig.update_yaxes(title_text="건수")
    
    return fig
```

#### 4.3.2 박스플롯 (대분류별 비교)

```python
def create_lag_boxplot(df: pd.DataFrame) -> go.Figure:
    """
    대분류별 Lag 박스플롯.
    """
    fig = px.box(
        df,
        x='대분류',
        y='Lag_일수',
        title='대분류별 Lag 분포',
        labels={'Lag_일수': 'Lag (일)', '대분류': '대분류'},
        color='대분류'
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig
```

### 4.4 UI 통합

**Step 6에 섹션 추가**:
```python
# Step 6-G: Lag 분석
st.write("#### ⏱️ Lag 분석 (제조일 ~ 접수일)")

# Lag 계산
lag_data = calculate_lag(filtered_claims)
lag_stats = analyze_lag(lag_data)

# 통계량 메트릭
col_lag1, col_lag2, col_lag3, col_lag4 = st.columns(4)

with col_lag1:
    st.metric("평균 Lag", f"{lag_stats['mean_lag']:.1f}일")

with col_lag2:
    st.metric("중앙값 Lag", f"{lag_stats['median_lag']:.1f}일")

with col_lag3:
    st.metric("최소 Lag", f"{lag_stats['min_lag']}일")

with col_lag4:
    st.metric("최대 Lag", f"{lag_stats['max_lag']}일")

# 히스토그램
fig_lag_hist = create_lag_histogram(lag_data)
st.plotly_chart(fig_lag_hist, use_container_width=True)

# 박스플롯 (대분류별)
if '대분류' in lag_data.columns:
    fig_lag_box = create_lag_boxplot(lag_data)
    st.plotly_chart(fig_lag_box, use_container_width=True)

# 이상 케이스 경고
if lag_stats['negative_count'] > 0:
    st.warning(
        f"⚠️ **음수 Lag 발견**: {lag_stats['negative_count']}건\n\n"
        "접수일이 제조일보다 이전입니다. 데이터 정합성을 확인해주세요."
    )
```

---

## 5. Data Mart 연동

### 5.1 현재 데이터 구조

**As-Is**:
```
data/
├── hub/                # 전체 클레임 데이터 (파티셔닝)
│   ├── 접수년=2024/
│   │   ├── 접수월=1/
│   │   │   └── part-0.parquet
│   │   ├── 접수월=2/
│   │   │   └── part-0.parquet
│   └── ...
├── sales/              # 매출 데이터
│   └── sales_history.parquet
└── models/             # 학습 모델
    └── {plant}_{major_category}/
        └── champion.pkl
```

**문제점**:
- 전체 데이터 로드 시 메모리 부담
- 특정 플랜트/대분류 분석 시에도 전체 스캔 필요

### 5.2 Data Mart 전략

#### 5.2.1 시리즈 분절 구조

**To-Be**:
```
data/
├── hub/                # 원본 데이터 (유지)
├── series/             # 분절 데이터 (신규)
│   ├── PlantA_관능_향.parquet
│   ├── PlantA_관능_맛.parquet
│   ├── PlantA_이물_금속.parquet
│   ├── PlantB_관능_향.parquet
│   └── ...
```

**파일명 규칙**:
```
{플랜트}_{대분류}_{소분류}.parquet
```

**장점**:
- 필요한 시리즈만 선택적 로드 → 메모리 절약
- 병렬 처리 용이 (각 시리즈 독립)
- 증분 업데이트 효율적 (변경된 시리즈만 갱신)

#### 5.2.2 시리즈 생성 함수

```python
# core/storage.py
def create_series_files(
    source_path: Union[str, Path] = DATA_HUB_PATH,
    target_dir: Union[str, Path] = 'data/series'
) -> None:
    """
    전체 데이터를 [플랜트 | 대분류 | 소분류] 단위로 분절하여 저장.
    
    Args:
        source_path: 원본 데이터 경로 (data/hub)
        target_dir: 시리즈 저장 디렉토리 (data/series)
    """
    # 전체 데이터 로드
    df = pd.read_parquet(source_path)
    
    # 시리즈 키 추출
    series_keys = df[['플랜트', '대분류', '소분류']].drop_duplicates()
    
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 각 시리즈별로 파일 생성
    for _, row in series_keys.iterrows():
        plant = row['플랜트']
        major = row['대분류']
        sub = row['소분류']
        
        # 해당 시리즈 데이터 필터링
        series_data = df[
            (df['플랜트'] == plant) &
            (df['대분류'] == major) &
            (df['소분류'] == sub)
        ]
        
        # 파일명 생성 (특수문자 제거)
        filename = f"{plant}_{major}_{sub}.parquet".replace('/', '_').replace(' ', '_')
        filepath = target_path / filename
        
        # 저장
        series_data.to_parquet(filepath, index=False)
        print(f"[SERIES] 생성: {filename} ({len(series_data)}건)")
    
    print(f"[SERIES] 총 {len(series_keys)}개 시리즈 생성 완료")
```

#### 5.2.3 시리즈 로드 함수

```python
def load_series(
    plant: str,
    major_category: str,
    sub_category: Optional[str] = None,
    series_dir: Union[str, Path] = 'data/series'
) -> pd.DataFrame:
    """
    특정 시리즈 데이터 로드.
    
    Args:
        plant: 플랜트명
        major_category: 대분류
        sub_category: 소분류 (None이면 대분류 전체)
        series_dir: 시리즈 디렉토리
    
    Returns:
        pd.DataFrame: 시리즈 데이터
    """
    series_path = Path(series_dir)
    
    if sub_category:
        # 특정 소분류 로드
        filename = f"{plant}_{major_category}_{sub_category}.parquet".replace('/', '_').replace(' ', '_')
        filepath = series_path / filename
        
        if filepath.exists():
            return pd.read_parquet(filepath)
        else:
            return pd.DataFrame()
    else:
        # 대분류 전체 로드 (여러 소분류 병합)
        pattern = f"{plant}_{major_category}_*.parquet"
        matching_files = list(series_path.glob(pattern))
        
        if matching_files:
            dfs = [pd.read_parquet(f) for f in matching_files]
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()
```

### 5.3 증분 업데이트

#### 5.3.1 트리거

**P1 데이터 업로드 시 자동 실행**:

```python
# pages/1_데이터_업로드.py (수정)
if st.button("💾 데이터 저장", type="primary"):
    # 기존: 파티셔닝 저장
    save_partitioned(processed_df, DATA_HUB_PATH)
    
    # 신규: 시리즈 파일 갱신
    with st.spinner("시리즈 파일 업데이트 중..."):
        create_series_files(DATA_HUB_PATH, 'data/series')
    
    st.success("✅ 데이터 저장 및 시리즈 업데이트 완료")
```

#### 5.3.2 스마트 업데이트 (선택적)

**변경된 시리즈만 갱신**:
```python
def update_series_incremental(
    new_data: pd.DataFrame,
    series_dir: Union[str, Path] = 'data/series'
) -> None:
    """
    변경된 시리즈만 증분 업데이트.
    """
    # 새 데이터의 시리즈 키 추출
    new_keys = new_data[['플랜트', '대분류', '소분류']].drop_duplicates()
    
    for _, row in new_keys.iterrows():
        plant, major, sub = row['플랜트'], row['대분류'], row['소분류']
        
        # 기존 시리즈 로드
        existing_data = load_series(plant, major, sub, series_dir)
        
        # 새 데이터 필터링
        new_series = new_data[
            (new_data['플랜트'] == plant) &
            (new_data['대분류'] == major) &
            (new_data['소분류'] == sub)
        ]
        
        # 병합 (중복 제거)
        combined = pd.concat([existing_data, new_series], ignore_index=True)
        combined = combined.drop_duplicates(subset=['상담번호'], keep='last')
        
        # 저장
        filename = f"{plant}_{major}_{sub}.parquet".replace('/', '_').replace(' ', '_')
        filepath = Path(series_dir) / filename
        combined.to_parquet(filepath, index=False)
        
        print(f"[UPDATE] {filename}: {len(new_series)}건 추가")
```

### 5.4 P3 연동

**Step 6 분석 시 시리즈 로드 활용**:
```python
# Step 6: 분석 시작
if st.button("🚀 분석 시작"):
    # 기존: 전체 데이터 로드
    # filtered_claims = st.session_state.claims_data[...]
    
    # 신규: 시리즈 기반 로드 (선택적 - 성능 개선 시)
    selected_series_data = []
    
    for major_cat in st.session_state.filter_major_category:
        series_df = load_series(
            plant=selected_plant,
            major_category=major_cat,
            sub_category=None,  # 대분류 전체
            series_dir='data/series'
        )
        selected_series_data.append(series_df)
    
    if selected_series_data:
        filtered_claims = pd.concat(selected_series_data, ignore_index=True)
    else:
        # Fallback: 전체 데이터 로드
        filtered_claims = st.session_state.claims_data[...]
```

---

## 6. 예측 엔진과의 접점

### 6.1 데이터 흐름

```
P3 (플랜트 분석)
    ├─ 필터링 조건 → P4로 전달
    ├─ 분석 결과 → 시리즈 메타데이터 저장
    └─ "P4에서 예측 실행" 버튼 → P4로 이동
         ↓
P4 (예측 시뮬레이션)
    ├─ P3에서 전달받은 필터 조건 자동 적용
    ├─ 시리즈 단위로 챔피언 모델 로드
    ├─ Seasonal Allocation 예측
    └─ 예측 결과 → P3로 반환 (피벗 테이블에 표시)
```

### 6.2 인터페이스 설계

#### 6.2.1 세션 상태 공유

```python
# P3 → P4 데이터 전달
if st.button("🔮 P4에서 예측 실행"):
    # P3의 필터 조건 저장
    st.session_state['p3_to_p4'] = {
        'plant': selected_plant,
        'major_categories': st.session_state.filter_major_category,
        'business_units': st.session_state.filter_business,
        'reasons': st.session_state.filter_reason,
        'grades': st.session_state.filter_grade,
        'pivot_rows': st.session_state.saved_pivot_rows
    }
    
    # P4로 이동
    st.switch_page("pages/4_예측_시뮬레이션.py")
```

#### 6.2.2 P4에서 필터 자동 적용

```python
# pages/4_예측_시뮬레이션.py
if 'p3_to_p4' in st.session_state:
    p3_config = st.session_state['p3_to_p4']
    
    st.info(f"✅ P3에서 전달받은 설정: {p3_config['plant']}")
    
    # 자동 선택
    selected_plant = p3_config['plant']
    selected_major_categories = p3_config['major_categories']
    
    # 예측 실행 (대분류별)
    for major_cat in selected_major_categories:
        predictions = predict_with_seasonal_allocation(
            plant=selected_plant,
            major_category=major_cat,
            future_months=[1, 2, 3],
            sub_dimensions_df=load_series(selected_plant, major_cat),
            model_dir='data/models'
        )
        
        # 결과 저장
        st.session_state[f'predictions_{major_cat}'] = predictions
```

### 6.3 예측 결과 P3 반영

#### 6.3.1 피벗 테이블에 예측 컬럼 추가

**현재**:
```
피벗 테이블:
| 대분류 | 중분류 | 2024-01 | 2024-02 | ... | 2024-12 |
```

**개선**:
```
피벗 테이블:
| 대분류 | 중분류 | 2024-01 | ... | 2024-12 | 2025-01(예측) | 2025-02(예측) | 2025-03(예측) |
```

**구현**:
```python
# create_pivot_table() 함수 수정
def create_pivot_table_with_forecast(
    df: pd.DataFrame,
    predictions_df: pd.DataFrame,  # 신규 인자
    index_cols: List[str],
    value_col: str = '건수'
) -> pd.DataFrame:
    """
    피벗 테이블 생성 (실적 + 예측 통합).
    """
    # 1. 실적 피벗 (기존)
    pivot_actual = create_pivot_table(df, index_cols, value_col)
    
    # 2. 예측 피벗
    if not predictions_df.empty:
        pivot_forecast = predictions_df.pivot_table(
            index=index_cols,
            columns='접수월',
            values='예측_건수',
            aggfunc='sum'
        )
        
        # 컬럼명 변경 (예측 표시)
        pivot_forecast.columns = [f"{col}(예측)" for col in pivot_forecast.columns]
        
        # 3. 병합 (실적 + 예측)
        pivot_combined = pd.concat([pivot_actual, pivot_forecast], axis=1)
        
        return pivot_combined
    else:
        return pivot_actual
```

#### 6.3.2 Step 6에 통합

```python
# Step 6-C: 예측 데이터 로드 (P4에서 생성된 경우)
predictions_df = pd.DataFrame()

for major_cat in st.session_state.filter_major_category:
    key = f'predictions_{major_cat}'
    if key in st.session_state:
        predictions_df = pd.concat([
            predictions_df,
            st.session_state[key]
        ], ignore_index=True)

# Step 6-E: 피벗 테이블 생성 (예측 포함)
if '건수' in st.session_state.selected_metrics:
    count_pivot = create_pivot_table_with_forecast(
        ppm_data,
        predictions_df,
        index_cols=pivot_index,
        value_col='건수'
    )
    st.dataframe(count_pivot)
```

---

## 7. 개발 일정

### 7.1 Phase 2 작업 분해 (WBS)

| 작업 ID | 작업명 | 예상 소요 시간 | 우선순위 |
|---------|--------|----------------|----------|
| **2.1** | **UI 재설계** | **4시간** | 높음 |
| 2.1.1 | 매크로 버튼 로직 확장 (core/config.py) | 1시간 | 높음 |
| 2.1.2 | Step 5 매크로 선택 UI 구현 | 1시간 | 높음 |
| 2.1.3 | Step 6 섹션 확장 (이상치, Lag) | 2시간 | 높음 |
| **2.2** | **이상치 감지** | **6시간** | 높음 |
| 2.2.1 | core/analytics.py 생성 | 0.5시간 | 높음 |
| 2.2.2 | detect_outliers_iqr() 구현 | 2시간 | 높음 |
| 2.2.3 | detect_outliers_zscore() 구현 | 1.5시간 | 중간 |
| 2.2.4 | 피벗 테이블 스타일링 함수 | 1시간 | 높음 |
| 2.2.5 | Step 6 이상치 섹션 UI | 1시간 | 높음 |
| **2.3** | **Lag 분석** | **5시간** | 높음 |
| 2.3.1 | calculate_lag() 함수 구현 | 1시간 | 높음 |
| 2.3.2 | analyze_lag() 통계량 함수 | 1시간 | 높음 |
| 2.3.3 | create_lag_histogram() 시각화 | 1.5시간 | 높음 |
| 2.3.4 | create_lag_boxplot() 구현 | 1시간 | 중간 |
| 2.3.5 | Step 6 Lag 섹션 UI | 0.5시간 | 높음 |
| **2.4** | **Data Mart 연동** | **8시간** | 중간 |
| 2.4.1 | create_series_files() 구현 | 2시간 | 중간 |
| 2.4.2 | load_series() 구현 | 1.5시간 | 중간 |
| 2.4.3 | update_series_incremental() 구현 | 2.5시간 | 낮음 |
| 2.4.4 | P1 업로드 페이지 트리거 연동 | 1시간 | 중간 |
| 2.4.5 | P3 시리즈 로드 통합 | 1시간 | 낮음 |
| **2.5** | **예측 연동** | **6시간** | 중간 |
| 2.5.1 | P3 → P4 세션 상태 전달 | 1.5시간 | 중간 |
| 2.5.2 | P4 자동 필터 적용 | 1.5시간 | 중간 |
| 2.5.3 | create_pivot_table_with_forecast() | 2시간 | 중간 |
| 2.5.4 | Step 6 예측 통합 UI | 1시간 | 중간 |
| **2.6** | **테스트 & 문서화** | **4시간** | 높음 |
| 2.6.1 | 단위 테스트 (analytics.py) | 2시간 | 높음 |
| 2.6.2 | 통합 테스트 (P3 전체 플로우) | 1.5시간 | 높음 |
| 2.6.3 | 사용자 가이드 작성 | 0.5시간 | 중간 |

**총 예상 시간**: **33시간** (약 4일)

### 7.2 일자별 계획

**D+1 (내일)**:
- [ ] 오전: 2.1 UI 재설계 (4시간)
- [ ] 오후: 2.2 이상치 감지 시작 (4시간) → 2.2.1~2.2.3 완료

**D+2**:
- [ ] 오전: 2.2 이상치 감지 완료 (2시간) → 2.2.4~2.2.5
- [ ] 오후: 2.3 Lag 분석 전체 (5시간)

**D+3**:
- [ ] 오전: 2.4 Data Mart 연동 (4시간) → 2.4.1~2.4.3
- [ ] 오후: 2.4 Data Mart 연동 완료 (2시간) → 2.4.4~2.4.5
- [ ] 저녁: 2.5 예측 연동 시작 (2시간) → 2.5.1~2.5.2

**D+4**:
- [ ] 오전: 2.5 예측 연동 완료 (3시간) → 2.5.3~2.5.4
- [ ] 오후: 2.6 테스트 & 문서화 (4시간)

---

## 8. 기술 스택

### 8.1 필수 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| **pandas** | 2.1.3 | 데이터 처리 |
| **numpy** | 1.24.3 | 수치 연산 (IQR, Z-Score) |
| **plotly** | 5.18.0 | 시각화 (히스토그램, 박스플롯) |
| **streamlit** | 1.28.1 | UI 프레임워크 |
| **pyarrow** | 14.0.1 | Parquet I/O |
| **scipy** | 1.11.4 | 통계 분석 (선택적) |

### 8.2 신규 파일 구조

```
core/
├── __init__.py
├── config.py               # PERFORMANCE_MACROS 추가
├── analytics.py            # 신규: 이상치, Lag 분석 함수
├── storage.py              # create_series_files, load_series 추가
├── etl.py
└── engine/
    ├── __init__.py
    ├── models.py
    └── trainer.py

data/
├── hub/                    # 기존
├── sales/                  # 기존
├── models/                 # 기존
└── series/                 # 신규: 시리즈 분절 데이터
    └── (동적 생성)

pages/
├── 1_데이터_업로드.py      # 시리즈 트리거 추가
├── 2_매출수량_관리.py      # 변경 없음
├── 3_플랜트_분석.py        # 대폭 수정 (Step 6 확장)
└── 4_예측_시뮬레이션.py    # P3 연동 추가

reports/
├── phase1_build_report.md
├── phase2_adaptive_report.md
├── phase2_development_plan.md  # 본 문서
├── phase3_optuna_report.md
├── phase3_5_fix_report.md
└── phase4_step1_engine.md
```

### 8.3 코드 품질 기준

| 항목 | 기준 |
|------|------|
| **Type Hinting** | 100% 적용 (모든 함수 인자 & 반환값) |
| **한글 주석** | 모든 함수에 docstring (한국어) |
| **에러 처리** | try-except 블록 + 사용자 친화적 메시지 |
| **로깅** | print() + st.info/warning/error |
| **테스트** | 핵심 함수 단위 테스트 (pytest) |

---

## 9. 리스크 관리

### 9.1 기술적 리스크

| 리스크 | 발생 가능성 | 영향도 | 완화 전략 |
|--------|-------------|--------|---------|
| **이상치 오탐** | 중간 | 높음 | 사용자가 IQR/Z-Score 선택 가능, 임계값 조정 UI 제공 |
| **Lag 데이터 누락** | 높음 | 중간 | dropna() 처리, 누락 개수 경고 표시 |
| **시리즈 파일 생성 시간** | 중간 | 중간 | 진행바 표시, 백그라운드 작업 (선택적) |
| **메모리 부족** | 낮음 | 높음 | 시리즈 단위 로드, 청크 처리 |
| **P3-P4 세션 상태 손실** | 낮음 | 중간 | 세션 상태 검증 로직 추가 |

### 9.2 일정 리스크

| 리스크 | 완화 전략 |
|--------|---------|
| **Data Mart 구현 지연** | 우선순위 낮춤 (Phase 3으로 연기 가능) |
| **예측 연동 복잡도** | 최소 기능(MVP)만 구현, 고급 기능은 Phase 4 |

---

## 10. 검증 시나리오

### 10.1 이상치 감지 테스트

**입력**:
```python
# 샘플 데이터
data = {
    '대분류': ['관능'] * 12,
    '중분류': ['향'] * 12,
    '년월': ['2024-01', '2024-02', ..., '2024-12'],
    '건수': [10, 12, 11, 13, 150, 9, 10, 11, 12, 10, 11, 10]  # 150이 이상치
}
```

**기대 결과**:
- IQR 방식: 150 탐지 (Upper Bound ≈ 20)
- 피벗 테이블에서 '2024-05' 열 빨간색 강조

### 10.2 Lag 분석 테스트

**입력**:
```python
# 샘플 데이터
data = {
    '제조일': ['2024-01-01', '2024-01-05', '2024-01-10'],
    '접수일': ['2024-01-15', '2024-01-06', '2024-01-08'],
    # Lag: [14일, 1일, -2일]
}
```

**기대 결과**:
- 평균 Lag: 4.3일
- 음수 Lag 경고: 1건

### 10.3 Data Mart 성능 테스트

**시나리오**: 100,000건 데이터, 10개 플랜트, 5개 대분류

**측정 지표**:
- 전체 로드 시간: ~10초
- 시리즈 로드 시간 (1개): ~0.5초
- 개선율: 95% 단축

---

## 11. 결론

본 개발 계획서는 **Phase 2: Deep Dive Analysis** 단계에서 구현할 4가지 핵심 기능에 대한 상세 설계를 제시합니다:

1. ✅ **P3 UI 재설계**: 6-Step 흐름에 이상치/Lag 섹션 추가, 매크로 확장
2. ✅ **이상치 감지**: IQR/Z-Score 알고리즘, 시각적 강조
3. ✅ **Lag 분석**: 제조~접수 시차 통계 및 히스토그램
4. ✅ **Data Mart 연동**: 시리즈 분절 전략으로 성능 최적화
5. ✅ **예측 연동**: P3 ↔ P4 데이터 흐름 설계

**예상 개발 기간**: 4일 (D+1 ~ D+4)  
**주요 위험 요소**: 이상치 오탐, 데이터 누락 처리  
**완화 전략**: 사용자 선택 가능한 알고리즘, 강건한 에러 처리

**다음 단계**: 내일(D+1)부터 실제 코드 구현 시작

---

**승인**: ___________________  
**날짜**: 2026-01-06

