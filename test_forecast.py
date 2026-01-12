#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '.')

from core.forecasting import ForecastEngine
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 간단한 테스트 데이터 (24개월 + 진행 중인 1월)
dates = pd.period_range('2024-01', periods=25, freq='M')
values = [1000 + 100*i + (i%12)*50 for i in range(25)]
series = pd.Series(values, index=dates)

# 이를 일일 데이터로 변환
daily_dates = []
daily_counts = []
for period, count in zip(dates[:-1], values[:-1]):
    start = period.start_time
    for day in range(30):
        daily_dates.append(start + pd.Timedelta(days=day))
        daily_counts.append(count // 30 + (1 if day < count % 30 else 0))

# 1월의 12일치만 추가 (진행 중)
jan_2026 = pd.Period('2026-01')
for day in range(1, 13):
    daily_dates.append(datetime(2026, 1, day))
    daily_counts.append(80 + day * 5)

df = pd.DataFrame({
    '접수일자': daily_dates,
    '건수': daily_counts
})

# ForecastEngine 초기화
print("🔍 ForecastEngine 초기화...")
engine = ForecastEngine(df, date_col='접수일자')

# 현재 날짜 (1월 12일)
current_date = datetime(2026, 1, 12)
current_val = sum([c for d, c in zip(daily_dates, daily_counts) if d.date() == current_date.date()])

print(f"현재값(1월 12일까지): {current_val}건")
print()

# 1. 당월 예측
print("=" * 70)
print("1️⃣ 당월(1월) 월말 예측")
print("=" * 70)
current_pred = engine.predict_current_month_advanced(current_val, current_date)
print(f"예측값: {current_pred['predicted_final']:,}건")
print(f"신뢰도: {current_pred['confidence']}")
print(f"진행률: {current_pred['progress']:.1f}%")
print(f"95% CI: [{current_pred['ci_lower']:,}, {current_pred['ci_upper']:,}]건")
print()

# 2. 3개월 예측
print("=" * 70)
print("2️⃣ 향후 3개월(2월~4월) 예측")
print("=" * 70)
future_preds = engine.predict_next_3_months()
method = future_preds.pop('method')
print(f"방식: {method}")
for month_str in sorted(future_preds.keys()):
    print(f"{month_str}월: {future_preds[month_str]:,}건")
print()

print("=" * 70)
print("✅ 테스트 완료: 당월과 3개월 예측이 분리됨!")
print("=" * 70)
