import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class ForecastAllocator:
    """
    [Phase 3.0 Core] Top-down Forecast Allocator
    
    Role:
    - 대분류 예측값(Total Forecast)의 '총량(Volume)' 정확도와
    - 소분류 예측값(Sub Forecasts)의 '패턴(Shape)' 정확도를 결합.
    - Algorithm: Proportional Reconciliation (비례 보정)
    
    Problem Solved:
    - 기존 '고정 비율(Static Ratio)' 방식이 소분류의 고유 계절성을 묵살하는 문제 해결.
    - 개별 예측 합계와 전체 예측 합계의 불일치(Mismatch)를 강제 보정하여 정합성 보장.
    """
    
    def __init__(self, total_df: pd.DataFrame, sub_dfs: Dict[str, pd.DataFrame]):
        """
        Args:
            total_df: 대분류 예측 결과 (Must have 'ds', 'y' columns)
            sub_dfs: 소분류별 예측 결과 Dict {'sub_name': df[['ds', 'y']]}
        """
        self.total_df = total_df.copy().sort_values('ds').reset_index(drop=True)
        self.sub_dfs = {k: v.copy().sort_values('ds').reset_index(drop=True) for k, v in sub_dfs.items()}
        
    def allocate(self) -> Dict[str, pd.DataFrame]:
        """
        실행 메인 함수 (Proportional Reconciliation)
        Logic: Sum(Sub) -> Mismatch Ratio (Total/Sum) -> Apply to Subs
        Returns: Dict[str, pd.DataFrame] (총량 보정이 완료된 소분류 예측 데이터)
        """
        # 1. 소분류 합계 계산 (Bottom-up Sum)
        combined_df = self.total_df[['ds']].copy()
        combined_df['ds'] = pd.to_datetime(combined_df['ds'])
        
        temp_cols = []
        for name, df in self.sub_dfs.items():
            df['ds'] = pd.to_datetime(df['ds'])
            merged = pd.merge(combined_df[['ds']], df[['ds', 'y']], on='ds', how='left').fillna(0)
            combined_df[f'sub_{name}'] = merged['y']
            temp_cols.append(f'sub_{name}')
            
        # 행별 합계 (Sum of Subs)
        combined_df['bottom_up_sum'] = combined_df[temp_cols].sum(axis=1)
        
        # 2. 대분류 타겟 병합 (Target Volume)
        target_df = self.total_df.copy()
        target_df['ds'] = pd.to_datetime(target_df['ds'])
        combined_df = pd.merge(combined_df, target_df[['ds', 'y']], on='ds', how='left')
        combined_df.rename(columns={'y': 'target_total'}, inplace=True)
        
        # 3. 보정 계수 산출 (Scaling Factor = Target / Sum)
        combined_df['scaling_factor'] = combined_df.apply(
            lambda x: x['target_total'] / x['bottom_up_sum'] if x['bottom_up_sum'] != 0 else 0, 
            axis=1
        )
        
        # 4. 각 소분류에 계수 적용 (Distribute)
        final_results = {}
        for name in self.sub_dfs.keys():
            original_sub = self.sub_dfs[name].copy()
            original_sub['ds'] = pd.to_datetime(original_sub['ds'])
            
            original_sub = pd.merge(original_sub, combined_df[['ds', 'scaling_factor']], on='ds', how='left')
            original_sub['scaling_factor'] = original_sub['scaling_factor'].fillna(1.0)
            
            original_sub['y'] = original_sub['y'] * original_sub['scaling_factor']
            final_results[name] = original_sub[['ds', 'y']]
            
        return final_results

    def validate_consistency(self, allocated_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """[Validation] 배분 후 총량 일치 여부 검증"""
        validation_df = self.total_df[['ds', 'y']].copy()
        validation_df.rename(columns={'y': 'Original_Total'}, inplace=True)
        validation_df['ds'] = pd.to_datetime(validation_df['ds'])
        
        temp_sum = pd.Series(0.0, index=validation_df.index)
        for name, df in allocated_results.items():
            df_sorted = df.sort_values('ds').reset_index(drop=True)
            temp_sum += df_sorted['y']
            
        validation_df['Allocated_Sum'] = temp_sum
        validation_df['Diff'] = validation_df['Original_Total'] - validation_df['Allocated_Sum']
        validation_df['Is_Matched'] = validation_df['Diff'].abs() < 1e-6
        return validation_df