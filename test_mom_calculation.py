"""
전월 동기 비(MoM - Month-on-Month 같은 날짜 비교) 계산 검증
현재: 2026-01-09
비교 대상: 2025-12-09 (정확한 전월 동기)
"""

import pandas as pd
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta

# 테스트 데이터 생성
data = {
    '접수일자': pd.date_range('2025-11-01', '2026-01-12', freq='D'),
    '건수': [10, 12, 15, 8, 9, 11, 13, 14, 16, 18, 20, 22, 25, 28, 30,  # 11월
             31, 29, 27, 25, 26, 28, 30, 32, 33, 35, 37, 39, 40, 42, 44, 45,  # 12월
             48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 75],  # 1월
    '등급기준': ['일반', '중대', '위험'] * 42
}

df = pd.DataFrame(data)
max_date = df['접수일자'].max()

print("=" * 80)
print("【전월 동기 비 계산 검증】")
print("=" * 80)
print(f"\n📅 기준 날짜: {max_date.strftime('%Y-%m-%d')} ({max_date.strftime('%A')})")
print(f"📊 테스트 데이터 범위: {df['접수일자'].min().strftime('%Y-%m-%d')} ~ {df['접수일자'].max().strftime('%Y-%m-%d')}")

# ============================================================
# [방법 1] 현재 코드 (오류)
# ============================================================
print("\n" + "=" * 80)
print("【방법 1: 현재 앱.py 코드 로직 (오류)】")
print("=" * 80)

day_of_month = max_date.day
current_month_start = max_date.replace(day=1)
prev_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
end_day_prev_month = min(day_of_month, pd.Timestamp(prev_month_start).days_in_month)
prev_month_end = prev_month_start.replace(day=end_day_prev_month)

print(f"\n📍 현재 범위: {current_month_start.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
print(f"📍 비교 범위: {prev_month_start.strftime('%Y-%m-%d')} ~ {prev_month_end.strftime('%Y-%m-%d')}")
print(f"   → 이것은 '누적 비교'입니다 (1월 1일~9일 vs 12월 1일~9일)")
print(f"   ⚠️  '전월 동기 비'가 아닙니다!")

df_current = df[(df['접수일자'] >= current_month_start) & (df['접수일자'] <= max_date)]
df_prev = df[(df['접수일자'] >= prev_month_start) & (df['접수일자'] <= prev_month_end)]

curr = df_current.shape[0]
past = df_prev.shape[0]
mom_wrong = ((curr - past) / past * 100) if past > 0 else 0

print(f"\n📊 결과:")
print(f"   현재 (2026-01-01~09): {curr}건")
print(f"   전월 (2025-12-01~09): {past}건")
print(f"   MoM 비율: {mom_wrong:+.1f}%")

# ============================================================
# [방법 2] 올바른 방식 - 정확한 전월 동기
# ============================================================
print("\n" + "=" * 80)
print("【방법 2: 올바른 전월 동기 비(Exact Same Date Comparison)】")
print("=" * 80)

# 현재 날짜: 2026-01-09
# 전월 동기: 2025-12-09 (정확히 1개월 전, 같은 일자)

current_date = max_date  # 2026-01-09
prev_month_same_date = current_date - relativedelta(months=1)  # 2025-12-09

print(f"\n📍 현재 날짜: {current_date.strftime('%Y-%m-%d')} ({current_date.strftime('%A')})")
print(f"📍 전월 동기: {prev_month_same_date.strftime('%Y-%m-%d')} ({prev_month_same_date.strftime('%A')})")

# 현재값: 2026-01-09 당일 건수
curr_value = df[df['접수일자'] == current_date].shape[0]
# 전월동기값: 2025-12-09 당일 건수
past_value = df[df['접수일자'] == prev_month_same_date].shape[0]

mom_correct_daily = ((curr_value - past_value) / past_value * 100) if past_value > 0 else 0

print(f"\n📊 "당일" 기준 MoM:")
print(f"   현재 ({current_date.strftime('%Y-%m-%d')}): {curr_value}건")
print(f"   전월 ({prev_month_same_date.strftime('%Y-%m-%d')}): {past_value}건")
print(f"   MoM 비율: {mom_correct_daily:+.1f}%")

# ============================================================
# [방법 3] 누적 기준 (월초~현재까지)
# ============================================================
print("\n" + "=" * 80)
print("【방법 3: 누적 기준 전월 동기 (월초~현재 누적)】")
print("=" * 80)

current_month_start_correct = current_date.replace(day=1)  # 2026-01-01
prev_month_start_correct = prev_month_same_date.replace(day=1)  # 2025-12-01

print(f"\n📍 현재 누적: {current_month_start_correct.strftime('%Y-%m-%d')} ~ {current_date.strftime('%Y-%m-%d')}")
print(f"📍 전월 누적: {prev_month_start_correct.strftime('%Y-%m-%d')} ~ {prev_month_same_date.strftime('%Y-%m-%d')}")

df_current_cumulative = df[(df['접수일자'] >= current_month_start_correct) & (df['접수일자'] <= current_date)]
df_prev_cumulative = df[(df['접수일자'] >= prev_month_start_correct) & (df['접수일자'] <= prev_month_same_date)]

curr_cumulative = df_current_cumulative.shape[0]
past_cumulative = df_prev_cumulative.shape[0]
mom_correct_cumulative = ((curr_cumulative - past_cumulative) / past_cumulative * 100) if past_cumulative > 0 else 0

print(f"\n📊 "누적" 기준 MoM:")
print(f"   현재 누적: {curr_cumulative}건")
print(f"   전월 누적: {past_cumulative}건")
print(f"   MoM 비율: {mom_correct_cumulative:+.1f}%")

# ============================================================
# 비교 요약
# ============================================================
print("\n" + "=" * 80)
print("【결과 비교】")
print("=" * 80)
print(f"\n방법 1 (현재 코드): {mom_wrong:+.1f}% (2026-01-01~09 vs 2025-12-01~09)")
print(f"방법 2 (당일 비교): {mom_correct_daily:+.1f}% (2026-01-09 vs 2025-12-09)")
print(f"방법 3 (누적 비교): {mom_correct_cumulative:+.1f}% (2026-01-01~09 vs 2025-12-01~09)")

print("\n" + "=" * 80)
print("【권장사항】")
print("=" * 80)
print("""
✅ "전월 동기 비"는 일반적으로 다음 중 하나를 의미합니다:

1️⃣  "당일 기준" (Daily YoY/MoM)
   - 비교: 2026-01-09 vs 2025-12-09 (정확히 1개월 전 같은 날)
   - 용도: 일간 변동성 추적

2️⃣  "월초~현재 누적" (Month-to-Date)
   - 비교: 2026-01-01~09 vs 2025-12-01~09 (각 월의 같은 기간)
   - 용도: 월간 진행 상황 모니터링

【현재 앱 문제점】
❌ 현재 코드는 전월의 "시작일(12/01)"을 기준으로 함
   → 2025-12-01 ~ 2025-12-09 범위와 비교
   → 실제 "전월 동기"가 아님

【권장 수정】
방법 3을 적용: prev_month_same_date를 기준으로 월초 계산
   prev_month_start = prev_month_same_date.replace(day=1)
""")
