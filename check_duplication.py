import pandas as pd
from pathlib import Path
import sys
import os

def verify_unique_claims(base_dir):
    """
    지정된 디렉토리 내의 모든 parquet 파일을 순회하며 상담번호 중복을 검사합니다.
    """
    # 경로 설정 (절대 경로 변환)
    root_path = Path(base_dir).resolve()
    print(f"🔍 검증 시작: {root_path} 내부의 파케이 파일을 스캔합니다...")

    # 모든 파케이 파일 재귀적 탐색
    parquet_files = list(root_path.rglob("*.parquet"))
    
    if not parquet_files:
        print("⚠️ 파케이 파일(.parquet)을 하나도 찾지 못했습니다.")
        print(f"   경로를 확인해주세요: {root_path}")
        return

    print(f"📦 총 {len(parquet_files)}개의 파일을 발견했습니다. 데이터 로딩 중...")

    # 데이터 수집
    combined_data = []
    
    for file_path in parquet_files:
        try:
            # 메모리 효율을 위해 '상담번호' 컬럼만 로드
            df = pd.read_parquet(file_path, columns=['상담번호'])
            
            # 출처 파일 추적을 위해 경로 정보 추가 (연도/월/파일명)
            # 예: 2022/1/part-0.parquet
            rel_path = file_path.relative_to(root_path)
            df['source_file'] = str(rel_path)
            
            combined_data.append(df)
            
        except Exception as e:
            print(f"❌ 읽기 실패 ({file_path.name}): {e}")

    if not combined_data:
        print("❌ 로드된 데이터가 없습니다.")
        return

    # 전체 데이터 병합
    full_df = pd.concat(combined_data, ignore_index=True)
    
    # 중복 검사 로직
    total_count = len(full_df)
    unique_count = full_df['상담번호'].nunique()
    duplicate_count = total_count - unique_count

    print("\n" + "="*40)
    print("📊 [검증 결과 요약]")
    print(f" - 전체 데이터 건수 : {total_count:,} 건")
    print(f" - 고유 상담번호 수 : {unique_count:,} 건")
    print(f" - 중복된 상담번호 수 : {duplicate_count:,} 건")
    print("="*40 + "\n")

    if duplicate_count > 0:
        print("🚨 **경고: 중복 데이터가 발견되었습니다!** 🚨\n")
        
        # 중복된 데이터 추출 (모든 중복 인스턴스 포함)
        duplicates = full_df[full_df.duplicated('상담번호', keep=False)].sort_values('상담번호')
        
        # 중복 상세 리포트
        dup_groups = duplicates.groupby('상담번호')['source_file'].apply(list)
        
        print("📄 [중복 상세 리포트]")
        for claim_id, sources in dup_groups.items():
            print(f"🔸 상담번호 [{claim_id}] ({len(sources)}회 중복):")
            for src in sources:
                print(f"   └─ {src}")
            print("-" * 30)
    else:
        print("✅ **검증 성공: 모든 상담번호가 유일합니다.** (무결성 확보)")

if __name__ == "__main__":
    # 기본 경로 설정 (사용자 환경에 맞춤)
    default_path = r"data/hub"
    
    # 터미널 인자로 경로를 받거나 기본값 사용
    target_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    if os.path.exists(target_path):
        verify_unique_claims(target_path)
    else:
        # 윈도우 절대 경로 예시로 재시도 (혹시 실행 위치가 다를 경우를 대비)
        abs_path = r"C:\claim-analysis-engine\data\hub"
        if os.path.exists(abs_path):
            verify_unique_claims(abs_path)
        else:
            print(f"❌ 경로를 찾을 수 없습니다: {target_path}")
            print("   사용법: python check_duplication.py [데이터폴더경로]")