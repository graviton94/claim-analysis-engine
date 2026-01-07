
---

### 📋 [Work Order] Pivot Table Visibility Improvement

**1. [Target File]**

* `pages/3_플랜트_분석.py`

**2. [Objective]**

* 피벗 테이블의 가로 길이가 지나치게 길어지는 것을 방지하기 위해 **"Hybrid View (과거 연평균 + 최근 월별 상세)"** 구조로 테이블 컬럼을 재구성한다.

**3. [Logic Requirements]**

**A. 기간 정의 (Period Splitting)**

* 사용자가 선택한 `end_date`를 기준으로 **최근 24개월(2년)**을 'Recent Period'로 정의한다.
* 그 이전 기간을 'Old Period'로 정의한다.
* *Example*: 종료일이 2024-03인 경우
* **Recent**: 2022-04 ~ 2024-03 (24개월)
* **Old**: 2022-03 이전 데이터



**B. 피벗 테이블 재구성 (Transformation)**

1. **Base Pivot**: 기존 로직대로 선택된 전체 기간의 월별 피벗 테이블(`pivot_base`)을 생성한다. (Zero-filling 유지)
2. **Old Period Aggregation (과거 요약)**:
* 'Old Period'에 해당하는 월 컬럼들을 연도별(`YYYY`)로 그룹핑한다.
* 각 연도별 **월평균(Mean)**을 계산한다. (소수점 1자리 반올림)
* 컬럼명: `YYYY년 Avg`


3. **Recent Period Display (최신 상세)**:
* 'Recent Period'에 해당하는 월 컬럼(`YYYY-MM`)은 그대로 유지한다.


4. **Summary Columns (우측 요약)**:
* 테이블 가장 우측(`Total` 왼쪽)에 다음 2개 컬럼을 추가한다.
* **`{직전년도}년 Avg`**: (예: 2023년 전체 합계 / 12)
* **`{당해년도}년 Avg`**: (예: 2024년 현재 합계 / 경과 월수)





**C. 최종 컬럼 순서 (Column Ordering)**

* `[Old Year Avg 컬럼들]` + `[Recent Monthly 컬럼들]` + `[직전년 Avg]` + `[당해년 Avg]` + `[Total]`
* **스타일링**: 평균 컬럼은 소수점 1자리(`1.5`), 월별 개수는 정수(`3`)로 포맷팅.

**4. [Implementation Hint for Agent]**

```python
# 힌트 로직 (의사 코드)
cutoff_date = end_date - relativedelta(months=23) # 최근 24개월 시작점

# 1. Base Pivot 생성 (기존 코드 활용)
df = create_pivot_with_subtotals(...) 

# 2. 컬럼 분류
old_cols = [c for c in df.columns if c < cutoff_str and c != 'Total']
recent_cols = [c for c in df.columns if c >= cutoff_str and c != 'Total']

# 3. Old Period -> 연도별 평균 변환
# (예: 2020-01~2020-12 -> '2020 Avg')

# 4. Summary Cols 계산
# last_year_avg = df[last_year_cols].mean(axis=1)
# this_year_avg = df[this_year_cols].mean(axis=1)

# 5. pd.concat으로 최종 DataFrame 조립

```

---
