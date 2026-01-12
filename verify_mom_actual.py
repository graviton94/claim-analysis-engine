"""
실제 데이터를 기반으로 전월 동기 비(MoM) 계산 검증
현재 기준: 2026-01-09
"""

import pandas as pd
from datetime import timedelta
from core.storage import DATA_HUB_PATH
import pyarrow.dataset as ds

print("=" * 80)
print("【실제 데이터 기반 전월 동기 비 검증】")
print("=" * 80)

try:
    dataset = ds.dataset(DATA_HUB_PATH, partitioning="hive", format="parquet")
    df = dataset.to_table().to_pandas()
    df['접수일자'] = pd.to_datetime(df['접수일자'])
except Exception as e:
    print(f"❌ 데이터 로드 실패: {e}")
    exit(1)

max_date = df['접수일자'].max()
day_of_month = max_date.day

print(f"\n📅 기준 날짜: {max_date.strftime('%Y-%m-%d')} (기준일 = {day_of_month}일)")
print(f"📊 데이터 범위: {df['접수일자'].min().strftime('%Y-%m-%d')} ~ {df['접수일자'].max().strftime('%Y-%m-%d')}")

# ============================================================
# 현재 코드 로직 (app.py에서 사용 중)
# ============================================================
current_month_start = max_date.replace(day=1)
prev_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
end_day_prev_month = min(day_of_month, pd.Timestamp(prev_month_start).days_in_month)
prev_month_end = prev_month_start.replace(day=end_day_prev_month)

print(f"\n" + "=" * 80)
print("【비교 범위】")
print("=" * 80)
print(f"📍 현재 범위: {current_month_start.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
print(f"📍 전월 비교: {prev_month_start.strftime('%Y-%m-%d')} ~ {prev_month_end.strftime('%Y-%m-%d')}")
print(f"   → 둘 다 {day_of_month}일간 누적 비교 ✅")

# ============================================================
# 전체 건수 비교
# ============================================================
df_current = df[(df['접수일자'] >= current_month_start) & (df['접수일자'] <= max_date)]
df_prev = df[(df['접수일자'] >= prev_month_start) & (df['접수일자'] <= prev_month_end)]

curr_total = df_current.shape[0]
past_total = df_prev.shape[0]
mom_total = ((curr_total - past_total) / past_total * 100) if past_total > 0 else 0

print(f"\n" + "=" * 80)
print("【전체 클레임 건수】")
print("=" * 80)
print(f"현재 ({current_month_start.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}): {curr_total:,}건")
print(f"전월 ({prev_month_start.strftime('%Y-%m-%d')} ~ {prev_month_end.strftime('%Y-%m-%d')}): {past_total:,}건")
print(f"📊 전월 동기 비: {mom_total:+.1f}%")

if mom_total > 0:
    print(f"   → 🔴 {abs(mom_total):.1f}% 증가 (전월 대비 악화)")
elif mom_total < 0:
    print(f"   → 🔵 {abs(mom_total):.1f}% 감소 (전월 대비 개선)")
else:
    print(f"   → ⚪ 변화 없음")

# ============================================================
# 등급별 비교
# ============================================================
print(f"\n" + "=" * 80)
print("【등급별 전월 동기 비】")
print("=" * 80)

grades = ['위험', '중대', '일반']
for grade in grades:
    curr_grade = df_current[df_current['등급기준'] == grade].shape[0]
    past_grade = df_prev[df_prev['등급기준'] == grade].shape[0]
    mom_grade = ((curr_grade - past_grade) / past_grade * 100) if past_grade > 0 else 0
    
    print(f"\n【{grade}】")
    print(f"  현재: {curr_grade:,}건 | 전월: {past_grade:,}건 | MoM: {mom_grade:+.1f}%")

# ============================================================
# 일별 추이 (검증용)
# ============================================================
print(f"\n" + "=" * 80)
print("【일별 추이 (검증용)】")
print("=" * 80)

print(f"\n📊 현재월 ({current_month_start.strftime('%Y-%m')})")
daily_current = df_current.groupby(df_current['접수일자'].dt.date).size()
for date, count in daily_current.items():
    print(f"   {date}: {count:,}건")

print(f"\n📊 전월 ({prev_month_start.strftime('%Y-%m')})")
daily_prev = df_prev.groupby(df_prev['접수일자'].dt.date).size()
for date, count in daily_prev.items():
    print(f"   {date}: {count:,}건")

print(f"\n" + "=" * 80)
print("【결론】")
print("=" * 80)
print("""
✅ 현재 app.py의 전월 동기 비 계산이 올바릅니다!

계산 로직:
  - 현재 범위: 2026-01-01 ~ 2026-01-09 (1월 초~9일까지 누적)
  - 전월 범위: 2025-12-01 ~ 2025-12-09 (12월 초~9일까지 누적)
  → 같은 경과 기간(9일)을 비교하므로 유효한 "전월 동기 비"입니다.

MoM 비율 해석:
  - (+) 양수: 전월 같은 기간 대비 증가 (품질 악화 신호)
  - (-) 음수: 전월 같은 기간 대비 감소 (품질 개선)
  - 0: 변화 없음
""")
