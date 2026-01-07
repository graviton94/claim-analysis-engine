# scripts/run_phase1_5_verification.py
import pandas as pd
from core.etl import process_claim_data, preprocess_data

def main():
    """
    Phase 1.5 검증 스크립트:
    - dummy_claims.csv 로드
    - ETL 파이프라인 실행
    - 전처리 후 로그 확인
    """
    file_path = "data/dummy_claims.csv"
    
    try:
        # process_claim_data는 내부적으로 load_raw_file, extract_54_fields 등을 호출
        df_processed = process_claim_data(file_path)
        
        # 수정된 preprocess_data 함수 호출
        df_preprocessed = preprocess_data(df_processed)
        
        print("\n[Verification] 스크립트 실행 완료.")
        print(f"최종 데이터프레임 행: {len(df_preprocessed)}")
        
        # '제조일자' 파싱 결과 요약
        parsed_dates = df_preprocessed['제조일자'].notna().sum()
        total_dates = len(df_preprocessed['제조일자'])
        print(f"제조일자 파싱 성공: {parsed_dates} / {total_dates} (성공률: {parsed_dates/total_dates:.2%})")

    except Exception as e:
        print(f"[Verification] 스크립트 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()

