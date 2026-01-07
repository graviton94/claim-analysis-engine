
## 🛠️ 작업 상세 지침 (Requirements)

### 1. ETL 전처리 로직 강화 (`core/etl.py`)

`preprocess_data(df)` 함수를 다음 규칙에 맞춰 고도화하라.

* **Type Casting**: `접수일자`, `제조일자`, `유통기한` 컬럼을 `pd.to_datetime(..., errors='coerce')`로 변환.
* **Lag Feature**:
* `Lag_Days` = (접수일자 - 제조일자).dt.days
* **Validation**: `Lag_Valid` 컬럼 생성. (조건: `Lag_Days >= 0` AND 날짜 컬럼 Not Null).
* 음수이거나 날짜 오류인 데이터는 삭제하지 말고 `Lag_Valid=False`로 마킹만 할 것.



### 2. Parquet 허브 저장 (`core/storage.py`)

* `save_partitioned_parquet(df)`: 기존 로직 유지하되, 위에서 생성된 `Lag_Days`, `Lag_Valid` 컬럼이 포함되어야 함.
* **Partitioning**: `접수년`, `접수월` 기준으로 폴더 구조화.

### 3. Nested Series JSON 생성 (`core/storage.py` - **Critical**)

`generate_nested_series(df)` 함수를 신규 작성하라.

* **Grouping Key**: `[플랜트, 제품범주2, 대분류]` (파일명 기준).
* **Zero-filling Logic**:
* 데이터셋 전체의 `Min Date` ~ `Max Date` 범위를 파악하여 모든 월(Month) 리스트를 생성.
* 실적이 없는 월은 `count: 0`으로 채워 시계열 연속성을 보장 (Parent & Children 모두).


* **JSON Schema (Nested)**:
```json
{
  "key": "{Plant}_{Cat2}_{Major}",
  "meta": {
    "last_updated": "YYYY-MM-DD",
    "warning_level": 0,           // 초기값 0
    "champion_model": null,       // 초기값 null
    "parent_stats": {             // 대분류 통계 (Lag_Valid=True인 데이터 기준)
      "mean": float, "std": float, "slope": float
    }
  },
  "data": {
    "history": [                  // 대분류 월별 실적 (Zero-filled)
      {"date": "YYYY-MM", "count": int}, ...
    ],
    "forecast": []                // 초기값 empty list
  },
  "children": [                   // 중분류 상세 데이터 리스트
    {
      "sub_key": "{Middle_Category}",
      "stats": {"mean": float, "std": float, "slope": float},
      "history": [                // 중분류 월별 실적 (Zero-filled)
        {"date": "YYYY-MM", "count": int}, ...
      ]
    },
    ...
  ]
}

```


* **Statistics**: 각 시리즈별 `mean`(평균), `std`(표준편차), `slope`(최근 3개월 기울기)를 계산하여 메타에 기록.

### 4. 업로드 페이지 연동 (`pages/1_데이터_업로드.py`)

[데이터 저장] 버튼 클릭 시 실행 순서를 다음과 같이 변경하라.

1. `preprocess_data` 실행 (데이터 정제).
2. `save_partitioned_parquet` 실행 (`data/hub` 갱신).
3. `generate_nested_series` 실행 (`data/series` 갱신).
4. `st.success` 메시지에 "Parquet 저장 및 OO개 Series JSON 생성 완료" 표시.

## 📝 검증 보고

작업 완료 후 `reports/phase1_refactor_nested.md`를 생성하여, 실제로 생성된 **Nested JSON 파일의 샘플 텍스트**를 출력하고 구조가 스키마와 일치하는지 보고하라.