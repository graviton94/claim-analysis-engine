# ============================================================================
# 페이지: 예측 시뮬레이션 (Optuna 챔피언 모델 기반)
# ============================================================================
# 설명: Optuna로 튜닝된 3개 모델을 학습하고, 우승 모델로 6개월 예측
#      성능 리더보드 및 시계열 시각화

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Tuple

from core.config import DATA_HUB_PATH, DATA_SALES_PATH, SALES_FILENAME
from core.storage import load_partitioned, load_sales_with_estimation
from core.engine.trainer import HyperParameterTuner, ChampionSelector

# ============================================================================
# 페이지 레이아웃 설정
# ============================================================================
st.set_page_config(page_title="예측 시뮬레이션", page_icon="🔮", layout="wide")
st.title("🔮 예측 시뮬레이션 (Optuna Champion Model)")
st.markdown(
    "CatBoost, SARIMAX, LSTM 3개 모델을 Optuna로 자동 튜닝하고, "
    "우승 모델의 6개월 예측 결과를 시각화합니다."
)

# ============================================================================
# 기본 설정
# ============================================================================
SALES_PATH = Path(DATA_SALES_PATH) / SALES_FILENAME
FORECAST_MONTHS = 6


# ============================================================================
# 함수: 데이터 준비
# ============================================================================

def prepare_timeseries_data(
    claims: pd.DataFrame,
    plant: str,
    product: Optional[str] = None,
    min_months: int = 12
) -> Tuple[pd.Series, Optional[pd.DataFrame], str]:
    """
    플랜트/제품군별 월별 클레임 건수 시계열 생성.
    
    Args:
        claims: 클레임 데이터
        plant: 플랜트명
        product: 제품군 (None이면 전체)
        min_months: 최소 데이터 개월 수
    
    Returns:
        Tuple: (y_series, exog_df, description)
    """
    # 플랜트 필터
    df = claims[claims['플랜트'] == plant].copy()
    
    # 제품군 필터 (optional)
    if product and product != "전체":
        df = df[df['제품군'] == product]
    
    # 월별 건수 집계
    df['연월'] = df['접수년'] * 100 + df['접수월']
    monthly = df.groupby(['접수년', '접수월', '연월']).size().reset_index(name='건수')
    monthly = monthly.sort_values(['접수년', '접수월']).reset_index(drop=True)
    
    # 시계열 확인
    if len(monthly) < min_months:
        raise ValueError(f"데이터 부족: {len(monthly)}개월 (최소 {min_months}개월 필수)")
    
    y_series = monthly['건수']
    
    # 외생변수 (매출수량) 로드
    try:
        sales = load_sales_with_estimation(SALES_PATH)
        sales_filtered = sales[sales['플랜트'] == plant].copy()
        
        # 월별 매출 추출
        exog_df = sales_filtered[['년', '월', '매출수량', 'is_estimated']].rename(
            columns={'년': '접수년', '월': '접수월'}
        ).sort_values(['접수년', '접수월']).reset_index(drop=True)
        
        # y_series와 길이 맞추기
        if len(exog_df) < len(y_series):
            # 부족한 행 추가 (NaN)
            missing = len(y_series) - len(exog_df)
            exog_df = pd.concat([
                pd.DataFrame({
                    '접수년': [monthly['접수년'].iloc[i] for i in range(missing)],
                    '접수월': [monthly['접수월'].iloc[i] for i in range(missing)],
                    '매출수량': [np.nan] * missing,
                    'is_estimated': [False] * missing
                }),
                exog_df
            ], ignore_index=True)
        else:
            exog_df = exog_df[:len(y_series)]
    
    except Exception as e:
        st.warning(f"⚠️ 매출 데이터 로드 실패: {str(e)}")
        exog_df = None
    
    # 설명 문자열
    description = f"{plant}"
    if product and product != "전체":
        description += f" - {product}"
    description += f" ({len(y_series)}개월)"
    
    return y_series, exog_df, description


# ============================================================================
# 세션 상태 초기화
# ============================================================================
if 'tuner' not in st.session_state:
    st.session_state.tuner = None
if 'selector' not in st.session_state:
    st.session_state.selector = None
if 'leaderboard' not in st.session_state:
    st.session_state.leaderboard = None
if 'forecast' not in st.session_state:
    st.session_state.forecast = None
if 'claims_data' not in st.session_state:
    st.session_state.claims_data = None


# ============================================================================
# 영역 1: 데이터 로드
# ============================================================================
st.subheader("📊 Step 1: 데이터 로드")

try:
    claims_data = load_partitioned(DATA_HUB_PATH)
    st.session_state.claims_data = claims_data
    st.success(f"✅ 클레임 데이터 로드: {len(claims_data)} 행")
except Exception as e:
    st.error(f"❌ 클레임 데이터 로드 실패: {str(e)}")
    st.stop()


# ============================================================================
# 영역 2: 플랜트/제품군 선택
# ============================================================================
st.subheader("🔍 Step 2: 분석 대상 선택")

col_plant, col_product = st.columns(2)

# 플랜트 선택
with col_plant:
    plants = sorted(st.session_state.claims_data['플랜트'].unique())
    selected_plant = st.selectbox("플랜트 선택 (필수)", plants, key="plant_select")

# 제품군 선택
with col_product:
    products = ['전체'] + sorted(
        st.session_state.claims_data[st.session_state.claims_data['플랜트'] == selected_plant]['제품군'].dropna().unique()
    )
    selected_product = st.selectbox("제품군 선택 (선택사항)", products, key="product_select")


# ============================================================================
# 영역 3: 학습 및 예측 프로세스
# ============================================================================
st.subheader("🚀 Step 3: 학습 및 예측")

col_tune, col_forecast = st.columns([2, 1])

with col_tune:
    n_trials = st.number_input("Optuna 시행 횟수", min_value=5, max_value=100, value=20, step=5)

with col_forecast:
    forecast_months = st.number_input("예측 기간 (개월)", min_value=1, max_value=12, value=6)

# 학습/예측 시작 버튼
if st.button("▶️ 학습 및 예측 시작", use_container_width=True, key="run_prediction"):
    
    # Progress 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: 데이터 준비
        status_text.info("📋 데이터 준비 중...")
        progress_bar.progress(10)
        
        y_series, exog_df, description = prepare_timeseries_data(
            st.session_state.claims_data,
            selected_plant,
            selected_product if selected_product != "전체" else None
        )
        
        st.info(f"분석 대상: {description}")
        
        # Step 2: Optuna 하이퍼파라미터 튜닝
        status_text.info("🔍 Optuna 하이퍼파라미터 튜닝 중 (SARIMAX, CatBoost, LSTM)...")
        progress_bar.progress(30)
        
        tuner = HyperParameterTuner(
            n_trials=n_trials,
            test_months=3,
            random_state=42
        )
        
        best_params = tuner.tune_all(y_series, exog=exog_df)
        st.session_state.tuner = tuner
        
        progress_bar.progress(60)
        
        # Step 3: 챔피언 선정 및 성능 비교
        status_text.info("🏆 챔피언 모델 선정 및 재학습 중...")
        progress_bar.progress(75)
        
        selector = ChampionSelector(best_params)
        leaderboard = selector.train_models(y_series, exog=exog_df, test_months=3)
        st.session_state.selector = selector
        st.session_state.leaderboard = leaderboard
        
        progress_bar.progress(85)
        
        # Step 4: 6개월 예측
        status_text.info("📈 6개월 예측 중...")
        progress_bar.progress(95)
        
        # 미래 외생변수 준비 (매출 없으면 NaN)
        future_exog = None
        if exog_df is not None:
            try:
                future_sales = exog_df[['매출수량']].tail(3).mean().values[0]
                future_exog = pd.DataFrame({
                    '매출수량': [future_sales] * forecast_months
                })
            except:
                future_exog = None
        
        forecast = selector.forecast(y_series, exog=future_exog, steps=forecast_months)
        st.session_state.forecast = forecast
        
        progress_bar.progress(100)
        status_text.success("✅ 학습 및 예측 완료!")
    
    except Exception as e:
        status_text.error(f"❌ 오류 발생: {str(e)}")
        st.stop()


# ============================================================================
# 영역 4: 결과 시각화
# ============================================================================

if st.session_state.leaderboard is not None and st.session_state.selector is not None:
    
    st.divider()
    st.subheader("📊 결과")
    
    # 4-1. 성능 리더보드
    st.write("#### 🏆 모델 성능 리더보드")
    
    leaderboard_display = st.session_state.leaderboard.copy()
    leaderboard_display['RMSE'] = leaderboard_display['RMSE'].round(2)
    leaderboard_display = leaderboard_display[['Rank', 'Model', 'RMSE']]
    
    # 선택된 행을 노란색으로 표시
    champion_name = st.session_state.selector.champion_name
    
    col1, col2, col3 = st.columns(3)
    
    for idx, row in leaderboard_display.iterrows():
        if row['Model'] == champion_name:
            with col1:
                st.metric(
                    f"🥇 {row['Model']} (Rank {row['Rank']})",
                    f"{row['RMSE']:.2f}",
                    delta="우승 모델"
                )
        elif row['Rank'] == 2:
            with col2:
                st.metric(
                    f"🥈 {row['Model']} (Rank {row['Rank']})",
                    f"{row['RMSE']:.2f}"
                )
        elif row['Rank'] == 3:
            with col3:
                st.metric(
                    f"🥉 {row['Model']} (Rank {row['Rank']})",
                    f"{row['RMSE']:.2f}"
                )
    
    st.dataframe(
        leaderboard_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Rank': st.column_config.NumberColumn('순위', format='%d'),
            'Model': st.column_config.TextColumn('모델'),
            'RMSE': st.column_config.NumberColumn('RMSE', format='%.2f')
        }
    )
    
    # 4-2. 예측 차트 (신뢰구간 포함)
    st.write("#### 📈 시계열 예측 (신뢰구간)")
    
    if st.session_state.forecast is not None:
        
        # 실제값 (최근 12개월)
        y_actual = st.session_state.claims_data[st.session_state.claims_data['플랜트'] == selected_plant]
        
        # 제품군 필터
        if selected_product != "전체":
            y_actual = y_actual[y_actual['제품군'] == selected_product]
        
        y_actual = y_actual.groupby(['접수년', '접수월']).size().reset_index(name='건수')
        y_actual = y_actual.sort_values(['접수년', '접수월']).reset_index(drop=True).tail(12)
        
        # 예측값
        y_forecast = st.session_state.forecast
        
        # 신뢰구간 (RMSE 기반)
        last_rmse = st.session_state.leaderboard.iloc[0]['RMSE']
        ci_upper = y_forecast + 1.96 * last_rmse
        ci_lower = np.maximum(y_forecast - 1.96 * last_rmse, 0)
        
        # 미래 날짜 생성
        last_year = st.session_state.claims_data['접수년'].max()
        last_month = st.session_state.claims_data[
            st.session_state.claims_data['접수년'] == last_year
        ]['접수월'].max()
        
        future_dates = []
        current_year = last_year
        current_month = last_month
        
        for _ in range(forecast_months):
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
            future_dates.append(f"{current_year}-{current_month:02d}")
        
        # Plotly 차트
        fig = go.Figure()
        
        # 실제값
        actual_dates = [
            f"{int(row['접수년'])}-{int(row['접수월']):02d}"
            for _, row in y_actual.iterrows()
        ]
        
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=y_actual['건수'].values,
            mode='lines+markers',
            name='실제값 (Actual)',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # 예측값
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=y_forecast,
            mode='lines+markers',
            name=f'예측값 ({champion_name})',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        # 신뢰구간
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(ci_upper) + list(ci_lower[::-1]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            showlegend=True,
            name='95% 신뢰구간'
        ))
        
        fig.update_layout(
            title=f"{description} - {champion_name} 모델 6개월 예측",
            xaxis_title="기간",
            yaxis_title="클레임 건수",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 예측값 테이블
        st.write("#### 📋 예측값 상세")
        
        forecast_df = pd.DataFrame({
            '기간': future_dates,
            '예측 건수': np.round(y_forecast, 1),
            '신뢰구간 하한': np.round(ci_lower, 1),
            '신뢰구간 상한': np.round(ci_upper, 1)
        })
        
        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                '기간': st.column_config.TextColumn('예측 기간'),
                '예측 건수': st.column_config.NumberColumn('예측값', format='%.1f'),
                '신뢰구간 하한': st.column_config.NumberColumn('95% CI (하한)', format='%.1f'),
                '신뢰구간 상한': st.column_config.NumberColumn('95% CI (상한)', format='%.1f')
            }
        )

else:
    st.info("💡 '학습 및 예측 시작' 버튼을 클릭하여 모델을 학습하고 예측 결과를 확인하세요.")
