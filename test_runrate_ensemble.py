#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '.')

from core.forecasting import ForecastEngine
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 테스트 데이터 생성 (24개월 + 진행 중인 1월)
dates = pd.period_range('2024-01', periods=25, freq='M')
# 각 월별로 다른 값 설정 (1월은 평소 900건 정도)
values = [
    1100, 1050, 900, 950, 1000, 1100,  # 2024-01 ~ 06
    1150, 1200, 1050, 980, 900, 1050,  # 2024-07 ~ 12
    1100, 1050, 900, 950, 1000, 1100,  # 2025-01 ~ 06
    1150, 1200, 1050, 980, 900, 1050,  # 2025-07 ~ 12
    0  # 2026-01 (진행 중, 나중에 덮어씀)
]
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
# 현재 페이스: 일별 100건 정도 (월말까지 가면 3000건 정도)
for day in range(1, 13):
    daily_dates.append(datetime(2026, 1, day))
    daily_counts.append(100)

df = pd.DataFrame({
    '접수일자': daily_dates,
    '건수': daily_counts
})

# ForecastEngine 초기화
print("=" * 80)
print("🔍 ForecastEngine 초기화...")
print("=" * 80)
engine = ForecastEngine(df, date_col='접수일자')

# 현재 날짜 (1월 12일)
current_date = datetime(2026, 1, 12)
current_val = 1200  # 1월 12일까지 1200건

print(f"현재값(1월 12일까지): {current_val:,}건")
print(f"진행률: 약 38.7% (12일 / 31일)")
print()

# ===== Run-rate 앙상블 상세 분석 =====
print("=" * 80)
print("🔬 Run-rate 앙상블 - 상세 분석")
print("=" * 80)

# 영업일 계산
month_start = datetime(2026, 1, 1)
current_date_obj = datetime(2026, 1, 12)
bdays_passed = int(pd.bdate_range(start=month_start, end=current_date_obj).size)
total_bdays = int(pd.bdate_range(start=datetime(2026, 1, 1), end=datetime(2026, 1, 31)).size)
progress = bdays_passed / total_bdays

print(f"경과 영업일: {bdays_passed}일")
print(f"전체 영업일: {total_bdays}일")
print(f"진행률: {progress:.1%}")
print()

# 1. 순수 Run-rate
daily_avg = current_val / bdays_passed
pred_runrate_raw = daily_avg * total_bdays
print(f"📊 순수 Run-rate:")
print(f"   = (현재값 / 경과일) × 전체일")
print(f"   = ({current_val} / {bdays_passed}) × {total_bdays}")
print(f"   = {pred_runrate_raw:,.0f}건")
print()

# 2. Back data (과거 1월 평균)
training_data = engine.training_series_cleaned
back_data_jan = training_data[training_data.index.month == 1]
if len(back_data_jan) > 0:
    back_data_avg = back_data_jan.mean()
    print(f"🏛️  Back data (과거 1월 평균):")
    print(f"   = {back_data_avg:,.0f}건")
    print(f"   (과거 {len(back_data_jan)}개 1월의 평균)")
else:
    back_data_avg = training_data.mean()
    print(f"🏛️  Back data (전체 평균):")
    print(f"   = {back_data_avg:,.0f}건")
print()

# 3. 가중치
if progress < 0.30:
    w_back = 0.70
    w_runrate = 0.30
elif progress < 0.70:
    w_back = 0.40
    w_runrate = 0.60
else:
    w_back = 0.20
    w_runrate = 0.80

print(f"⚖️  진행률 {progress:.1%}에 따른 가중치:")
print(f"   Back data: {w_back:.0%} (과거 패턴)")
print(f"   Run-rate: {w_runrate:.0%} (실시간 페이스)")
print()

# 4. 앙상블
pred_runrate_ensemble = (w_back * back_data_avg) + (w_runrate * pred_runrate_raw)
print(f"✅ Run-rate 앙상블:")
print(f"   = {w_back:.0%} × {back_data_avg:,.0f} + {w_runrate:.0%} × {pred_runrate_raw:,.0f}")
print(f"   = {pred_runrate_ensemble:,.0f}건")
print()

# ===== 당월 예측 =====
print("=" * 80)
print("1️⃣ 당월(1월) 월말 예측")
print("=" * 80)
current_pred = engine.predict_current_month_advanced(current_val, current_date)
print(f"예측값: {current_pred['predicted_final']:,}건")
print(f"신뢰도: {current_pred['confidence']}")
print(f"진행률: {current_pred['progress']:.1f}%")
print(f"95% CI: [{current_pred['ci_lower']:,}, {current_pred['ci_upper']:,}]건")
print()

# 모델별 예측
models = current_pred.get('models', {})
print(f"📊 모델별 예측:")
print(f"   Run-rate 앙상블: {models.get('runrate', 0):,}건")
print(f"   통계 기대치: {models.get('stat_base', 0):,}건")
print()

# ===== 3개월 예측 =====
print("=" * 80)
print("2️⃣ 향후 3개월(2월~4월) 예측")
print("=" * 80)
future_preds = engine.predict_next_3_months()
method = future_preds.pop('method')
print(f"방식: {method}")
for month_str in sorted(future_preds.keys()):
    print(f"{month_str}월: {future_preds[month_str]:,}건")
print()

print("=" * 80)
print("✅ 테스트 완료!")
print("=" * 80)
