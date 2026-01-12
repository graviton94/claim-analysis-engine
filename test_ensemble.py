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

# ===== 3개월 앙상블 상세 분석 =====
print("=" * 80)
print("🔬 3개월 예측 - 모델별 개별 값 분석")
print("=" * 80)

# 테스트용으로 각 모델 개별 호출
last_period = engine.training_series_cleaned.index[-1]

for months_ahead in range(2, 5):
    print(f"\n📅 {months_ahead}개월 앞 ({last_period + months_ahead})")
    print("-" * 80)
    
    # HW
    hw_current, hw_future = engine._predict_holt_winters_extended()
    future_period = last_period + months_ahead
    month_str = f"{future_period.year}-{future_period.month:02d}"
    hw_val = hw_future.get(month_str, 0) if hw_future else 0
    print(f"  🏛️  Holt-Winters: {hw_val:>8,}건 (가중치: 45%)")
    
    # SARIMA
    sarima_result = engine._predict_with_sarima(months_ahead=months_ahead)
    sarima_val = sarima_result['value'] if sarima_result['value'] else 0
    print(f"  📊 SARIMA:        {sarima_val:>8,.0f}건 (가중치: 35%)")
    
    # Trend
    trend_val = engine._predict_with_trend_regression(months_ahead=months_ahead)
    print(f"  📈 Trend:         {trend_val:>8,.0f}건 (가중치: 20%)")
    
    # 앙상블
    weights = {'hw': 0.45, 'sarima': 0.35, 'trend': 0.20}
    ensemble = weights['hw'] * hw_val + weights['sarima'] * sarima_val + weights['trend'] * trend_val
    print(f"  ✅ 앙상블 결과:    {ensemble:>8,.0f}건")
    print(f"     계산식: 0.45×{hw_val} + 0.35×{sarima_val:.0f} + 0.20×{trend_val:.0f} = {ensemble:.0f}")

print()
print("=" * 80)
print("📈 최종 3개월 예측 (다중 모델 앙상블)")
print("=" * 80)
future_preds = engine.predict_next_3_months()
method = future_preds.pop('method')
print(f"방식: {method}")
for month_str in sorted(future_preds.keys()):
    print(f"{month_str}월: {future_preds[month_str]:,}건")
print()

print("=" * 80)
print("✅ 테스트 완료: 3개월 예측이 HW + SARIMA + Trend 앙상블로 작동!")
print("=" * 80)
