import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
from io import BytesIO
import pyarrow.dataset as ds
import pyarrow as pa

# Core Engine Loading
from core.config import DATA_HUB_PATH
from core.engine.trainer import SimulationEngine

# ==============================================================================
# 1. 페이지 설정
# ==============================================================================
st.set_page_config(
    page_title="예측 시뮬레이션 Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧪 Prediction Simulation Lab (v6.1)")
st.markdown("""
이 실험실에서는 **Deep Learning & AutoML** 기술을 활용하여 정밀한 미래 예측을 수행합니다.
1. **Competition**: 3가지 모델(Prophet, AutoML, SARIMAX)이 경합하여 미래 추세를 예측합니다.
2. **Allocation**: 예측된 대분류 총량을 과거 패턴에 기반하여 **소분류로 자동 배분**합니다.
""")

# ==============================================================================
# 2. 데이터 로드 (Lightweight for UI)
# ==============================================================================
@st.cache_data(ttl=3600)
def load_metadata():
    """UI 구동용 경량 데이터 로드 (필수 컬럼만 읽음)"""
    try:
        import os
        if not os.path.exists(DATA_HUB_PATH):
            st.error(f"데이터 경로를 찾을 수 없습니다: {DATA_HUB_PATH}")
            return pd.DataFrame()

        dataset = ds.dataset(DATA_HUB_PATH, format="parquet", partitioning="hive")
        
        # UI 필터링 및 차트 표시에 꼭 필요한 컬럼만 로드
        cols = ['플랜트', '대분류', '소분류', '접수일자', '건수', '불만원인', '사업부문', '등급기준']
        available_cols = dataset.schema.names
        read_cols = [c for c in cols if c in available_cols]
        
        df = dataset.to_table(columns=read_cols).to_pandas()
        
        if '접수일자' in df.columns:
            df['접수일자'] = pd.to_datetime(df['접수일자'])
            
        if '건수' not in df.columns:
            df['건수'] = 1
            
        return df
        
    except Exception as e:
        st.error(f"데이터 로딩 실패: {str(e)}")
        return pd.DataFrame()

# [NEW] 다운로드용 전체 데이터 로드 함수
def load_full_target_data(plant, major, mode):
    """선택된 대상의 '모든 컬럼'을 원본에서 다시 읽어옴"""
    try:
        dataset = ds.dataset(DATA_HUB_PATH, format="parquet", partitioning="hive")
        
        # 플랜트 기준으로만 필터링 (대분류가 여러 개일 수 있으므로)
        if major and major != "All":
            majors = [m.strip() for m in major.split(",")]
            table = dataset.to_table()
            df_full = table.to_pandas()
            
            # 플랜트로 필터링
            df_full = df_full[df_full['플랜트'] == plant]
            
            # 선택된 대분류로 필터링
            if majors and majors[0]:  # 리스트가 비어있지 않고 첫 번째 요소가 비어있지 않음
                df_full = df_full[df_full['대분류'].isin(majors)]
        else:
            # 플랜트 기준만 필터링
            table = dataset.to_table()
            df_full = table.to_pandas()
            df_full = df_full[df_full['플랜트'] == plant]
        
        if '접수일자' in df_full.columns:
            df_full['접수일자'] = pd.to_datetime(df_full['접수일자'])
        
        # 모드에 따른 필터링 적용
        if mode == "실적 (Performance)":
            if '불만원인' in df_full.columns:
                reasons = ['고객불만족', '구매불만', '제조불만']
                df_full = df_full[df_full['불만원인'].isin(reasons)]
            
            if '사업부문' in df_full.columns:
                biz_units = ['식품', 'B2B식품']
                df_full = df_full[df_full['사업부문'].isin(biz_units)]
                
        return df_full
    except Exception as e:
        st.warning(f"백데이터 로드 경고: {e}")
        return pd.DataFrame()

with st.spinner("💾 데이터베이스 로딩 중..."):
    df_raw = load_metadata()

if df_raw.empty:
    st.error("❌ 데이터가 없습니다. [1. 데이터 업로드] 페이지에서 데이터를 먼저 적재해주세요.")
    st.stop()

# ==============================================================================
# 3. 사이드바: 실험 파라미터만 표시
# ==============================================================================
st.sidebar.header("📊 실험 파라미터")
forecast_months = st.sidebar.slider("예측 기간 (개월)", 3, 12, 6)
n_trials = st.sidebar.number_input("AutoML 시도 횟수", 10, 50, 15, help="높을수록 정확하지만 느려집니다.")

# ==============================================================================
# 4. 메인 화면: Step 1~3 필터링 로직 (3_플랜트_분석과 동일)
# ==============================================================================

# [CONFIG] 필터링 기준값
TARGET_BUSINESS_UNITS = ['식품', 'B2B식품']
PERFORMANCE_REASONS = ['제조불만', '고객불만족', '구매불만']

# --- Step 1: 범위 설정 (플랜트 + 기간) ---
st.markdown("#### Step 1: 분석 범위 설정")
col_s1_1, col_s1_2, col_s1_3 = st.columns([1, 1, 1])

all_plants = sorted(df_raw['플랜트'].dropna().unique()) if not df_raw.empty else []

with col_s1_1:
    sel_plant = st.selectbox("🏭플랜트 선택", all_plants, key='sim_plant_select')

# 플랜트별 데이터 범위 감지
plant_data = df_raw[df_raw['플랜트'] == sel_plant] if sel_plant else pd.DataFrame()
if not plant_data.empty:
    min_dt = plant_data['접수일자'].min()
    max_dt = plant_data['접수일자'].max()
    min_date = min_dt.replace(day=1).date()
    next_month = max_dt.replace(day=1) + pd.Timedelta(days=32)
    next_month = next_month.replace(day=1)
    max_date = (next_month - pd.Timedelta(days=1)).date()
else:
    min_date = df_raw['접수일자'].min().date() if not df_raw.empty else None
    max_date = df_raw['접수일자'].max().date() if not df_raw.empty else None

with col_s1_2:
    start_date = st.date_input("📅시작일 (Start)", value=min_date, min_value=min_date, max_value=max_date, key='sim_start_date')

with col_s1_3:
    end_date = st.date_input("📅종료일 (End)", value=max_date, min_value=min_date, max_value=max_date, key='sim_end_date')

# Step 1 필터링 결과
if sel_plant and start_date and end_date:
    plant_df = df_raw[
        (df_raw['플랜트'] == sel_plant) &
        (df_raw['접수일자'].dt.date >= start_date) &
        (df_raw['접수일자'].dt.date <= end_date)
    ].copy()
    st.info(f"📋 **요약**: `{sel_plant}` | `{start_date} ~ {end_date}` | 대상 **{len(plant_df):,}** 건")
else:
    plant_df = pd.DataFrame()
    st.warning("플랜트와 기간을 선택해주세요.")

st.divider()

# --- Step 2 & 3: 조회 모드 및 필터 ---
col_step2, col_step3 = st.columns(2)

with col_step2:
    st.markdown("#### Step 2: 조회 모드")
    
    def reset_sim_custom_selections():
        if 'sim_sel_biz' in st.session_state: 
            del st.session_state['sim_sel_biz']
        if 'sim_sel_reason' in st.session_state: 
            del st.session_state['sim_sel_reason']
    
    search_mode = st.radio(
        "조회 모드를 선택하세요:",
        ("인입 (Inflow)", "실적 (Performance)", "Custom (직접 선택)"),
        horizontal=True,
        key='sim_search_mode',
        on_change=reset_sim_custom_selections
    )
    
    filtered_df_step2 = plant_df.copy()
    
    # 모드별 필터링
    if search_mode == "인입 (Inflow)":
        cond_biz = filtered_df_step2['사업부문'].isin(TARGET_BUSINESS_UNITS)
        cond_reason = filtered_df_step2['불만원인'].notna()
        filtered_df_step2 = filtered_df_step2[cond_biz & cond_reason]
        st.caption(f"ℹ️ **인입 기준**: 사업부문({', '.join(TARGET_BUSINESS_UNITS)}) + 불만원인(전체)")
        
    elif search_mode == "실적 (Performance)":
        cond_biz = filtered_df_step2['사업부문'].isin(TARGET_BUSINESS_UNITS)
        cond_reason = filtered_df_step2['불만원인'].isin(PERFORMANCE_REASONS)
        filtered_df_step2 = filtered_df_step2[cond_biz & cond_reason]
        st.caption(f"ℹ️ **실적 기준**: 사업부문({', '.join(TARGET_BUSINESS_UNITS)}) + 불만원인({', '.join(PERFORMANCE_REASONS)})")
        
    else:  # Custom
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            opts_biz = sorted(plant_df['사업부문'].dropna().unique())
            sel_biz = st.multiselect("사업부문 선택", opts_biz, key='sim_sel_biz')
        with col_c2:
            opts_reason = sorted(plant_df['불만원인'].dropna().unique())
            sel_reason = st.multiselect("불만원인 선택", opts_reason, key='sim_sel_reason')
        
        if sel_biz:
            filtered_df_step2 = filtered_df_step2[filtered_df_step2['사업부문'].isin(sel_biz)]
        if sel_reason:
            filtered_df_step2 = filtered_df_step2[filtered_df_step2['불만원인'].isin(sel_reason)]

with col_step3:
    st.markdown("#### Step 3: 등급, 대분류 필터")
    
    # === 1. 등급 필터 ===
    grade_options = sorted(filtered_df_step2['등급기준'].dropna().unique())
    if 'sim_step3_grades' not in st.session_state:
        st.session_state['sim_step3_grades'] = grade_options
    
    selected_grades = st.multiselect(
        "분석할 등급을 선택하세요:",
        grade_options,
        key='sim_step3_grades'
    )
    
    # 등급 선택 여부 체크
    if not selected_grades:
        filtered_df_for_category = filtered_df_step2.copy()
    else:
        filtered_df_for_category = filtered_df_step2[filtered_df_step2['등급기준'].isin(selected_grades)].copy()
    
    # === 2. 대분류 필터 (등급에 따라 필터링됨) ===
    st.markdown("")
    category_options = sorted(filtered_df_for_category['대분류'].dropna().unique())
    
    if 'sim_step3_categories' not in st.session_state:
        st.session_state['sim_step3_categories'] = category_options
    
    selected_categories = st.multiselect(
        "분석할 대분류를 선택하세요:",
        category_options,
        key='sim_step3_categories'
    )
    
    # === 3. 최종 필터링 적용 ===
    df_target = filtered_df_step2.copy()
    
    if selected_grades:
        df_target = df_target[df_target['등급기준'].isin(selected_grades)]
    
    if selected_categories:
        df_target = df_target[df_target['대분류'].isin(selected_categories)]
    
    cnt_step3 = len(df_target)
    st.caption(f"📊 필터링 후 대상 건수: **{cnt_step3:,}** 건")

st.divider()

# ==============================================================================
# 5. Step 4: 시뮬레이션 실행 및 결과 표시
# ==============================================================================

# Action Button & Session State
if 'sim_results' not in st.session_state:
    st.session_state['sim_results'] = None

def run_simulation():
    st.session_state['run_clicked'] = True
    st.session_state['sim_results'] = None

btn_run = st.button("🚀 시뮬레이션 시작", type="primary", use_container_width=True, disabled=df_target.empty, on_click=run_simulation)

st.divider()

# [C] 시뮬레이션 실행 및 결과 표시 (Session State 기반)
# 버튼을 눌렀거나, 이미 결과가 저장되어 있는 경우 실행
if (st.session_state.get('run_clicked') or st.session_state.get('sim_results')) and not df_target.empty:
    st.divider()
    
    # --- 계산 로직 (결과가 없을 때만 실행) ---
    if st.session_state['sim_results'] is None:
        try:
            with st.spinner("🧠 AI 모델 엔진 가동 중... (예측 및 백데이터 로딩)"):
                # [Step 4-1] 필터된 데이터 aggregation (월별 합계로 변환)
                if '건수' not in df_target.columns: 
                    df_target['건수'] = 1
                
                # 월별 클레임 건수 집계
                monthly_counts = df_target.groupby(df_target['접수일자'].dt.to_period('M')).size()
                monthly_df = pd.DataFrame({
                    'ds': monthly_counts.index.to_timestamp(),
                    'y': monthly_counts.values
                })
                
                # [Step 4-2] 예측 엔진 실행 (aggregated 데이터 사용)
                if len(monthly_df) < 3:
                    st.error("예측을 위해 최소 3개월 이상의 데이터가 필요합니다.")
                    st.stop()
                
                engine = SimulationEngine(monthly_df, date_col='ds', val_col='y')
                df_forecast = engine.run_competition(periods=forecast_months)
                
                # [Step 4-3] 배분 로직 실행 (선택된 대분류 기반)
                sel_major_list = selected_categories if selected_categories else []
                sel_major_str = ", ".join(sel_major_list) if sel_major_list else "All"
                
                alloc_df = engine.predict_with_allocation(
                    plant=sel_plant,
                    major_category=sel_major_str,
                    sub_df=df_target,
                    periods=forecast_months,
                    forecast_df=df_forecast
                )

                # [Step 4-4] 다운로드용 전체 컬럼 데이터 로드 (Heavy Task)
                df_full_backdata = load_full_target_data(sel_plant, sel_major_str, search_mode)
                
                # [Step 4-5] 결과 저장
                st.session_state['sim_results'] = {
                    'engine': engine,
                    'df_forecast': df_forecast,
                    'alloc_df': alloc_df,
                    'df_full_backdata': df_full_backdata,
                    'model_weights': engine.model_weights,
                    'monthly_df': monthly_df
                }
        except Exception as e:
            st.error(f"시뮬레이션 중 오류 발생: {str(e)}")
            st.stop()
            
    # --- 결과 시각화 (Session State에서 로드) ---
    results = st.session_state['sim_results']
    engine = results['engine']
    df_forecast = results['df_forecast']
    alloc_df = results['alloc_df']
    df_full_backdata = results['df_full_backdata']
    monthly_df = results['monthly_df']
    
    # --- 1. 제목 및 조건 표시 ---
    selected_grades_str = ", ".join(selected_grades) if selected_grades else "전체 등급"
    st.subheader(f"📈 시뮬레이션 결과 ({selected_grades_str} / {search_mode})")
    
    # --- 2. 모델 경합 그래프 (Full Width) ---
    st.markdown(f"#### 🏁 모델 경합 (Model Competition)")
    
    # 그래프 데이터 준비
    recent_hist = engine.train_data.tail(12)
    if not recent_hist.empty:
        start_month_graph = recent_hist.index[0].strftime('%Y-%m')
        end_month_graph = monthly_df['ds'].max().strftime('%Y-%m')
    else:
        start_month_graph = monthly_df['ds'].min().strftime('%Y-%m')
        end_month_graph = monthly_df['ds'].max().strftime('%Y-%m')
    
    st.markdown(f"📊 2개년 추이 분석 ({start_month_graph} ~ {end_month_graph})")
    
    # 그래프 시각화
    fig_pred = go.Figure()
    
    # 1. 과거 데이터
    if not recent_hist.empty:
        last_date = recent_hist.index[-1]
        last_val = recent_hist.values[-1]
        
        fig_pred.add_trace(go.Scatter(
            x=recent_hist.index, y=recent_hist.values,
            mode='lines+markers', name='실적 History',
            line=dict(color='black', width=2),
            marker=dict(size=6)
        ))
    else:
        last_date = monthly_df['ds'].iloc[-1]
        last_val = monthly_df['y'].iloc[-1]
    
    # 2. 예측 데이터
    model_styles = {
        'Ensemble': {'color': '#6200ea', 'width': 5, 'dash': 'solid'},
        'AutoML':   {'color': '#e74c3c', 'width': 2, 'dash': 'dot'},
        'Prophet':  {'color': '#2ecc71', 'width': 1, 'dash': 'dot'},
        'SARIMAX':  {'color': '#3498db', 'width': 1, 'dash': 'dot'}
    }
    
    cols_sorted = [c for c in df_forecast.columns if c != 'Ensemble']
    if 'Ensemble' in df_forecast.columns:
        cols_sorted.append('Ensemble')
    
    for model_name in cols_sorted:
        style = model_styles.get(model_name, {'color': 'gray', 'width': 1, 'dash': 'dot'})
        
        # Gap Filling
        pred_x = [last_date] + list(df_forecast.index)
        pred_y = [last_val] + list(df_forecast[model_name].values)
        
        mode_style = 'lines+markers' if model_name == 'Ensemble' else 'lines'
        
        fig_pred.add_trace(go.Scatter(
            x=pred_x, y=pred_y,
            mode=mode_style,
            name=f'{model_name} 예측',
            line=dict(color=style['color'], width=style['width'], dash=style['dash']),
            opacity=1.0 if model_name == 'Ensemble' else 0.5
        ))

    fig_pred.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
        plot_bgcolor='white'
    )
    fig_pred.add_vline(x=last_date, line_width=1, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # 모델 가중치 (간단 표시)
    weights = results['model_weights']
    if weights:
        weights_text = " | ".join([f"{model}: {w*100:.1f}%" for model, w in weights.items()])
        st.info(f"⚖️ **모델별 가중치**: {weights_text}")

    if 'Ensemble' in df_forecast.columns:
        avg_pred = df_forecast['Ensemble'].mean()
        st.success(f"🏆 최종 앙상블(Ensemble) 예측 결과, 향후 월평균 **{avg_pred:.0f}건**이 예상됩니다.")
    
    st.divider()

    # --- 4. Risk 현황진단 ---
    st.markdown("#### 🛡️ Risk 현황진단")
    
    try:
        # 과거 12개월 + 예측 데이터 조합
        past_12_months = monthly_df.tail(12).copy() if len(monthly_df) >= 12 else monthly_df.copy()
        
        # 예측 데이터 포함
        if 'Ensemble' in df_forecast.columns:
            forecast_with_dates = pd.DataFrame({
                'ds': df_forecast.index,
                'y': df_forecast['Ensemble'].values
            })
            combined_df = pd.concat([past_12_months, forecast_with_dates], ignore_index=True)
        else:
            combined_df = past_12_months.copy()
        
        # 간단한 위험 신호 판단 (변화율 기반)
        if len(combined_df) >= 2:
            combined_df['change'] = combined_df['y'].pct_change() * 100
            combined_df['signal'] = '⚪'
            
            # Red: 급증 (>50% 증가) or 급감 (<-30%)
            combined_df.loc[(combined_df['change'] > 50) | (combined_df['change'] < -30), 'signal'] = '🔴'
            # Yellow: 중증도 증가 (20-50%) or 중증도 감소 (-30 ~ -10%)
            combined_df.loc[((combined_df['change'] > 20) & (combined_df['change'] <= 50)) | 
                           ((combined_df['change'] >= -30) & (combined_df['change'] < -10)), 'signal'] = '🟡'
        else:
            combined_df['signal'] = '⚪'
        
        # 경보 집계
        red_count = (combined_df['signal'] == '🔴').sum()
        yellow_count = (combined_df['signal'] == '🟡').sum()
        
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.markdown(f"##### Red(🔴) 경보: {red_count}건")
            red_df = combined_df[combined_df['signal'] == '🔴']
            if red_df.empty:
                st.info("경보 대상이 없습니다.")
            else:
                st.dataframe(red_df[['ds', 'y', 'change', 'signal']], use_container_width=True)
        
        with c_right:
            st.markdown(f"##### Yellow(🟡) 주의: {yellow_count}건")
            yellow_df = combined_df[combined_df['signal'] == '🟡']
            if yellow_df.empty:
                st.info("주의 대상이 없습니다.")
            else:
                st.dataframe(yellow_df[['ds', 'y', 'change', 'signal']], use_container_width=True)
        
        st.caption("위험 신호는 월별 변화율 기준으로 판정됩니다. (과거 12개월 + 예측 데이터 포함)")
    
    except Exception as e:
        st.warning(f"Risk 현황진단 생성 중 오류: {e}")
    
    st.divider()

    # --- 5. 피벗 테이블, Lag분석, 원본데이터 탭 ---
    tab1, tab2, tab3 = st.tabs(["피벗 테이블", "Lag 분석", "원본 데이터"])
    
    with tab1:
        st.markdown("##### 📊 과거 12개월 + 예측 데이터 피벗 테이블")
        
        try:
            # 과거 12개월 + 예측 데이터 조합 (소분류 기준 피벗)
            cutoff_date = pd.to_datetime(end_date) - relativedelta(months=12)
            historical_12m = df_target[df_target['접수일자'] >= cutoff_date].copy()
            historical_12m['월'] = historical_12m['접수일자'].dt.strftime('%Y-%m')
            
            if not historical_12m.empty:
                pivot_hist = pd.pivot_table(
                    historical_12m,
                    index='소분류',
                    columns='월',
                    values='상담번호',
                    aggfunc='count',
                    fill_value=0
                )
                
                # 예측 데이터 추가
                alloc_pivot = alloc_df.pivot_table(
                    index='소분류',
                    columns='예측월',
                    values='예측건수',
                    aggfunc='sum',
                    fill_value=0
                )
                
                # 결합
                combined_pivot = pd.concat([pivot_hist, alloc_pivot], axis=1)
                combined_pivot = combined_pivot.fillna(0)
                combined_pivot.loc['Total'] = combined_pivot.sum()
                
                st.dataframe(
                    combined_pivot.style
                        .format("{:,.0f}")
                        .background_gradient(cmap="Blues", subset=(combined_pivot.index[:-1], combined_pivot.columns)),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("과거 12개월 데이터가 없습니다.")
        
        except Exception as e:
            st.warning(f"피벗 테이블 생성 오류: {e}")
    
    with tab2:
        st.markdown("##### ⏱️ Lag 분석 (제조 ~ 접수 소요기간)")
        
        try:
            # Lag 계산 (접수일자 기준)
            if '접수일자' in df_target.columns:
                df_target['접수일_만'] = df_target['접수일자'].dt.date
                lag_stats_data = df_target.groupby('접수일_만').size()
                
                if len(lag_stats_data) > 0:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("평균 일일 클레임", f"{lag_stats_data.mean():.1f}")
                    c2.metric("중앙값", f"{lag_stats_data.median():.1f}")
                    c3.metric("최대값", f"{lag_stats_data.max():.0f}")
                    
                    # 분포 그래프
                    fig_lag = px.histogram(
                        df_target.groupby('접수일_만').size().reset_index(name='count'),
                        x='count',
                        nbins=30,
                        title="일일 클레임 건수 분포"
                    )
                    st.plotly_chart(fig_lag, use_container_width=True)
                else:
                    st.info("Lag 데이터가 없습니다.")
            else:
                st.warning("접수일자 컬럼이 없습니다.")
        
        except Exception as e:
            st.warning(f"Lag 분석 생성 오류: {e}")
    
    with tab3:
        st.markdown("##### 📋 필터된 원본 데이터")
        st.dataframe(df_target, use_container_width=True, height=500)

elif not btn_run and not st.session_state.get('run_clicked'):
    st.info("👈 위의 Step 1 ~ 3을 설정하고 [시뮬레이션 시작] 버튼을 눌러주세요.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Top-down Forecasting")
        st.caption("대분류 단위로 노이즈를 줄이고 거시적인 추세를 먼저 예측합니다.")
    with c2:
        st.markdown("#### 2. Model Competition")
        st.caption("Prophet, LightGBM, SARIMAX가 경합하여 최적의 예측선을 찾습니다.")
    with c3:
        st.markdown("#### 3. Seasonal Allocation")
        st.caption("총 예측량을 과거 패턴에 맞춰 제품(소분류) 단위로 정교하게 쪼갭니다.")