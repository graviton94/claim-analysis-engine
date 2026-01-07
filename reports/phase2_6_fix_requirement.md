
---

### 📋 [Hotfix] Pivot Table MultiIndex Error Fix

**1. [Target File]**

* `pages/3_플랜트_분석.py`

**2. [Function to Fix]**

* `create_pivot_with_subtotals`

**3. [Logic Requirements]**
이 함수를 아래의 **Robust Logic**으로 전면 교체하라. 인덱스 개수(`n_levels`)에 따라 소계 계산 깊이를 자동으로 조절해야 한다.

* **Case 1 (n=1)**: 기본 피벗에 `Total` 행만 추가하여 반환.
* **Case 2 (n=2)**: Level 2 소계 루프를 건너뛰고, **Level 1(상위 그룹) 합계**와 **Grand Total**만 계산.
* **Case 3 (n>=3)**: 기존처럼 **Level 2 소계**와 **Level 1 합계**, **Grand Total**을 모두 계산.

**4. [Replacement Code]**
(이 코드는 인덱스 길이에 상관없이 작동하는 안전한 버전이다. 복사해서 사용하라.)

```python
        def create_pivot_with_subtotals(df, indices, columns, values, aggfunc, all_months):
            # 1. Base Pivot
            pivot_base = pd.pivot_table(df, index=indices, columns=columns, values=values, aggfunc=aggfunc, fill_value=0)
            
            # Zero-filling
            pivot_base = pivot_base.reindex(columns=all_months, fill_value=0)
            
            if pivot_base.empty:
                empty_idx = pd.MultiIndex.from_tuples([], names=indices)
                return pd.DataFrame(0, index=empty_idx, columns=all_months + ['Total'])

            # 2. Grand Total Calculation (Common)
            grand_total_series = pivot_base.sum()
            grand_total_series.name = "Total"
            
            # 3. Dynamic Subtotal Logic
            n_levels = len(indices)
            
            # Case A: 인덱스가 1개인 경우 -> 소계 불필요, 총계만 붙여서 리턴
            if n_levels == 1:
                pivot_base['Total'] = pivot_base.sum(axis=1)
                # Grand Total Row
                grand_total_df = grand_total_series.to_frame('Total').T
                grand_total_df.index = pd.Index(['Total'], name=indices[0])
                return pd.concat([pivot_base, grand_total_df])

            # Case B: 인덱스가 2개 이상인 경우 -> 소계 계산
            all_parts = []
            
            # Level 0 (최상위) 기준으로 순회
            for l1_name, l1_group in pivot_base.groupby(level=0, sort=False):
                
                # --- [Logic] Level 2 소계 (인덱스가 3개 이상일 때만 수행) ---
                if n_levels >= 3:
                    for l2_name, l2_group in l1_group.groupby(level=1, sort=False):
                        all_parts.append(l2_group) # 원본 데이터 추가
                        
                        # 소계 행 생성
                        subtotal_row = l2_group.sum().to_frame().T
                        
                        # 인덱스 튜플 생성: (L1, L2, '소계', '', ...)
                        idx_parts = [l1_name, l2_name, '소계'] + [''] * (n_levels - 3)
                        subtotal_row.index = pd.MultiIndex.from_tuples([tuple(idx_parts)], names=indices)
                        all_parts.append(subtotal_row)
                else:
                    # 인덱스가 2개뿐이면, 그냥 원본 그룹을 통째로 추가 (L2 소계 없음)
                    all_parts.append(l1_group)

                # --- [Logic] Level 1 합계 (항상 수행) ---
                total_l1_row = l1_group.sum().to_frame().T
                
                # 인덱스 튜플 생성: (L1, '전체 합계', '', ...)
                idx_parts = [l1_name, '전체 합계'] + [''] * (n_levels - 2)
                total_l1_row.index = pd.MultiIndex.from_tuples([tuple(idx_parts)], names=indices)
                all_parts.append(total_l1_row)
            
            final_pivot = pd.concat(all_parts)
            
            # 4. Grand Total Row Append
            grand_total_df = grand_total_series.to_frame('Total').T
            # 인덱스 튜플 생성: ('Total', '', '', ...)
            idx_parts = ['Total'] + [''] * (n_levels - 1)
            grand_total_df.index = pd.MultiIndex.from_tuples([tuple(idx_parts)], names=indices)
            
            final_pivot = pd.concat([final_pivot, grand_total_df])
            
            # 5. Calculate Right-side Total Column
            final_pivot['Total'] = final_pivot[all_months].sum(axis=1)

            return final_pivot

```