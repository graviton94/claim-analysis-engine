import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pyarrow.dataset as ds

# [Core Module Import]
# Code B와 동일한 모듈을 사용하여 정합성 확보
from core.config import DATA_HUB_PATH
from core.storage import load_and_filter_data, get_claim_keys
from core.engine.trainer import SimulationEngine
from core.analytics import calculate_advanced_risk_score, calculate_lag_stats, prepare_risk_data, ForecastRiskAnalyzer

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

# [CONFIG] 필터링 기준값
TARGET_BUSINESS_UNITS = ['식품', 'B2B식품']
PERFORMANCE_REASONS = ['제조불만', '고객불만족', '구매불만']

# --- 2. 초기 데이터 로드 (플랜트 목록) ---
@st.cache_data(ttl=3600)
def load_plant_list():
    """초기 진입 시 플랜트 목록만 가볍게 로드"""
    keys = get_claim_keys(DATA_HUB_PATH)
    if not keys.empty:
        return sorted(keys['플랜트'].unique().tolist())
    return []

@st.cache_data(show_spinner=False)
def get_plant_date_range(plant):
    """플랜트별 데이터의 실제 날짜 범위 조회"""
    try:
        from datetime import date as date_type
        df = load_and_filter_data(
            plant=plant,
            start_date=date_type(2000, 1, 1),  # 매우 오래된 시작
            end_date=date_type(2100, 12, 31),  # 미래까지
            search_mode="Raw"  # 필터링 없음
        )
        if df.empty:
            return None, None
        
        min_date = df['접수일자'].min().date()
        max_date = df['접수일자'].max().date()
        return min_date, max_date
    except:
        return None, None

all_plants = load_plant_list()

if not all_plants:
    st.error("❌ 데이터가 없습니다. [1. 데이터 업로드] 페이지에서 데이터를 먼저 적재해주세요.")
    st.stop()

# ==============================================================================
# 2-1. 쿼리 파라미터 처리 (Risk Radar "예측" 버튼에서 전달)
# ==============================================================================
if st.query_params and 'plant' in st.query_params:
    qp_mode_raw = st.query_params.get('mode', '인입')
    qp_plant = st.query_params['plant']
    qp_grade = st.query_params.get('grade', '')
    qp_category = st.query_params.get('category', '')
    
    # Mode 매핑 (쿼리 파라미터 형식 → 라디오 옵션 형식)
    mode_map = {'인입': '인입', '실적': '실적', 'Raw': 'Raw data'}
    qp_mode = mode_map.get(qp_mode_raw, '인입')
    
    # 쿼리 파라미터 고유키 생성 (중복 처리 방지)
    qp_key = f"{qp_mode}|{qp_plant}|{qp_grade}|{qp_category}"
    
    # 마지막으로 적용한 쿼리 파라미터와 다르면 업데이트
    if st.session_state.get('last_sim_qp_key') != qp_key:
        st.session_state['sim_plant_select'] = qp_plant if qp_plant in all_plants else all_plants[0]
        st.session_state['sim_search_mode'] = qp_mode
        
        # 등급 필터: 파라미터로 받은 등급만 선택
        if qp_grade:
            st.session_state['sim_step3_grades'] = [qp_grade]
        
        # 대분류 필터: 파라미터로 받은 대분류만 선택
        if qp_category:
            st.session_state['sim_step3_categories'] = [qp_category]
        
        st.session_state['last_sim_qp_key'] = qp_key

# ==============================================================================
# 3. 사이드바: 실험 파라미터
# ==============================================================================
st.sidebar.header("📊 실험 파라미터")
forecast_months = st.sidebar.slider("예측 기간 (개월)", 3, 12, 6)
n_trials = st.sidebar.number_input("AutoML 시도 횟수", 10, 50, 15, help="높을수록 정확하지만 느려집니다.")

# ==============================================================================
# 4. 메인 화면: Step 1~3 필터링 (Code B와 로직 통일)
# ==============================================================================

# --- Step 1: 범위 설정 (플랜트 + 기간) ---
st.markdown("#### Step 1: 분석 범위 설정")
col_s1_1, col_s1_2, col_s1_3 = st.columns([1, 1, 1])

def on_plant_change():
    # 플랜트 변경 시 하위 필터 초기화
    keys_to_clear = ['sim_sel_biz', 'sim_sel_reason', 'sim_step3_grades', 'sim_step3_categories']
    for k in keys_to_clear:
        if k in st.session_state: del st.session_state[k]
    # 결과 초기화
    if 'sim_results' in st.session_state: st.session_state['sim_results'] = None
    if 'run_clicked' in st.session_state: st.session_state['run_clicked'] = False
    
    # 플랜트 변경 시 감지된 날짜 미리 저장 (에러 상황에 대비)
    selected = st.session_state.get('sim_plant_select')
    if selected:
        min_date, max_date = get_plant_date_range(selected)
        if min_date and max_date:
            st.session_state['detected_sim_start_date'] = min_date
            st.session_state['detected_sim_end_date'] = max_date
    # 선택된 플랜트의 날짜 범위 조회 및 업데이트
    plant = st.session_state.get('sim_plant_select')
    if plant:
        min_date, max_date = get_plant_date_range(plant)
        if min_date and max_date:
            st.session_state['detected_sim_start_date'] = min_date
            st.session_state['detected_sim_end_date'] = max_date

with col_s1_1:
    sel_plant = st.selectbox("🏭플랜트 선택", all_plants, key='sim_plant_select', on_change=on_plant_change)

# 플랜트 선택 시 자동으로 감지된 날짜 범위 사용, 아니면 기본값
today = datetime.today().date()
last_year_jan1 = date(today.year - 1, 1, 1)

default_start = st.session_state.get('detected_sim_start_date', last_year_jan1)
default_end = st.session_state.get('detected_sim_end_date', today)

with col_s1_2:
    start_date = st.date_input("📅시작일 (Start)", value=default_start)

with col_s1_3:
    end_date = st.date_input("📅종료일 (End)", value=default_end)

# 옵션 갱신용 데이터 로드 (가볍게)
@st.cache_data(show_spinner=False)
def get_options_data(plant, s_date, e_date):
    return load_and_filter_data(plant, s_date, e_date, search_mode="Custom")

with st.spinner("옵션 로딩 중..."):
    df_for_options = get_options_data(sel_plant, start_date, end_date)

if df_for_options.empty:
    st.warning(f"선택한 조건 ({sel_plant}, {start_date}~{end_date})에 해당하는 데이터가 없습니다.")
    # 에러 발생 시: 플랜트의 전체 범위 안내
    min_date, max_date = st.session_state.get('detected_sim_start_date'), st.session_state.get('detected_sim_end_date')
    if min_date and max_date:
        st.info(f"💡 **해당 플랜트의 데이터 범위**: {min_date} ~ {max_date}")
        st.info(f"👉 위의 시작일/종료일을 **{min_date}** ~ **{max_date}**로 변경해주세요.")
    st.stop()
else:
    st.info(f"📋 **요약**: `{sel_plant}` | `{start_date} ~ {end_date}` | 대상 **{len(df_for_options):,}** 건")

st.divider()

# --- Step 2 & 3: 조회 모드 및 필터 ---
col_step2, col_step3 = st.columns(2)

# 필터 변수 초기화
sim_sel_biz = []
sim_sel_reason = []
sim_sel_grades = []
sim_sel_categories = []

with col_step2:
    st.markdown("#### Step 2: 조회 모드")

    search_mode = st.radio(
        "조회 모드를 선택하세요:",
        ("인입", "실적", "Raw data"),
        horizontal=True,
        key='sim_search_mode'
    )

    if search_mode == "인입":
        st.caption(f"📋 사업부문: {', '.join(TARGET_BUSINESS_UNITS)} | 불만유형: 전체")
    elif search_mode == "실적":
        st.caption(f"📋 사업부문: {', '.join(TARGET_BUSINESS_UNITS)} | 불만유형: {', '.join(PERFORMANCE_REASONS)}")
    else: # Raw data
        st.caption("📋 CIS 기준 상담구분: 클레임 전체")

with col_step3:
    st.markdown("#### Step 3: 등급, 대분류 필터")
    
    # 1. 등급 필터
    grade_options = sorted(df_for_options['등급기준'].dropna().unique())
    
    # 세션 상태 초기화 (첫 진입 또는 플랜트 변경 시)
    if 'sim_step3_grades' not in st.session_state:
        st.session_state['sim_step3_grades'] = grade_options
    
    # 세션 상태의 값이 현재 옵션에 없으면 초기화
    current_grades = st.session_state.get('sim_step3_grades', grade_options)
    current_grades = [g for g in current_grades if g in grade_options]
    if not current_grades:
        current_grades = grade_options
        st.session_state['sim_step3_grades'] = current_grades
    
    # key가 있으면 default 파라미터를 사용하지 말 것 (Streamlit이 자동으로 세션 상태에서 읽음)
    sim_sel_grades = st.multiselect(
        "분석할 등급을 선택하세요:", 
        grade_options, 
        key='sim_step3_grades'
    )
    
    # 2. 대분류 필터
    st.markdown("")
    temp_df = df_for_options
    if sim_sel_grades:
        temp_df = temp_df[temp_df['등급기준'].isin(sim_sel_grades)]
        
    category_options = sorted(temp_df['대분류'].dropna().unique())
    
    # 세션 상태 초기화 (첫 진입 또는 플랜트 변경 시)
    if 'sim_step3_categories' not in st.session_state:
        st.session_state['sim_step3_categories'] = category_options
    
    # 세션 상태의 값이 현재 옵션에 없으면 초기화
    current_categories = st.session_state.get('sim_step3_categories', category_options)
    current_categories = [c for c in current_categories if c in category_options]
    if not current_categories:
        current_categories = category_options
        st.session_state['sim_step3_categories'] = current_categories
    
    # key가 있으면 default 파라미터를 사용하지 말 것
    sim_sel_categories = st.multiselect(
        "분석할 대분류를 선택하세요:", 
        category_options, 
        key='sim_step3_categories'
    )
    
    # 카운트 미리보기
    # (여기서는 정확한 필터링 카운트보다 UX 흐름이 중요하므로 생략하거나 df_for_options 기반 추정 가능)

st.divider()

# ==============================================================================
# 5. Step 4: 시뮬레이션 실행 및 결과 표시
# ==============================================================================

if 'sim_results' not in st.session_state:
    st.session_state['sim_results'] = None

def run_simulation():
    st.session_state['run_clicked'] = True
    st.session_state['sim_results'] = None

# 실행 가능 여부 체크
can_run = not df_for_options.empty
btn_run = st.button("🚀 시뮬레이션 시작", type="primary", width='stretch', disabled=not can_run, on_click=run_simulation)

st.divider()

# [C] 시뮬레이션 실행 로직
if (st.session_state.get('run_clicked') or st.session_state.get('sim_results')):
    
    # --- 계산 (결과 없을 때만) ---
    if st.session_state['sim_results'] is None:
        try:
            with st.spinner("🧠 AI 모델 엔진 가동 중... (데이터 로드 및 학습)"):
                # 1. Data Loading via Core (Single Source of Truth)
                # A. Target Data (Display & Training)
                df_target = load_and_filter_data(
                    plant=sel_plant,
                    start_date=start_date,
                    end_date=end_date,
                    search_mode=search_mode,
                    selected_biz=sim_sel_biz,
                    selected_reasons=sim_sel_reason,
                    selected_grades=sim_sel_grades,
                    selected_categories=sim_sel_categories
                )
                
                # B. Risk Data (History 24M for Zero-Filling)
                risk_start_date = start_date - relativedelta(months=24)
                df_risk_raw = load_and_filter_data(
                    plant=sel_plant,
                    start_date=risk_start_date,
                    end_date=end_date,
                    search_mode=search_mode,
                    selected_biz=sim_sel_biz,
                    selected_reasons=sim_sel_reason,
                    selected_grades=sim_sel_grades,
                    selected_categories=sim_sel_categories
                )
                
                if df_target.empty:
                    st.error("조건에 맞는 데이터가 없어 예측을 수행할 수 없습니다.")
                    st.stop()

                # 2. Data Aggregation for Training
                if '건수' not in df_target.columns: df_target['건수'] = 1
                
                monthly_counts = df_target.groupby(df_target['접수일자'].dt.to_period('M')).size()
                monthly_df = pd.DataFrame({
                    'ds': monthly_counts.index.to_timestamp(),
                    'y': monthly_counts.values
                })
                
                if len(monthly_df) < 3:
                    st.error("예측을 위해 최소 3개월 이상의 데이터가 필요합니다.")
                    st.stop()
                
                # 3. Prediction Engine Run
                engine = SimulationEngine(monthly_df, date_col='ds', val_col='y')
                df_forecast = engine.run_competition(periods=forecast_months)
                
                # 4. Allocation Logic
                sel_major_list = sim_sel_categories if sim_sel_categories else []
                sel_major_str = ", ".join(sel_major_list) if sel_major_list else "All"
                
                alloc_df = engine.predict_with_allocation(
                    plant=sel_plant,
                    major_category=sel_major_str,
                    sub_df=df_target,
                    periods=forecast_months,
                    forecast_df=df_forecast
                )
                
                # 5. Risk Data Prep (Zero-Filling) for Tab 1
                # 고정된 컬럼 기준: 등급, 대분류, 소분류
                fixed_pivot_indices = ['등급기준', '대분류', '소분류']
                risk_pivot_df = prepare_risk_data(
                    df=df_risk_raw,
                    pivot_keys=fixed_pivot_indices,
                    target_date=end_date,
                    lookback_months=24
                )
                
                # 6. Save Results
                st.session_state['sim_results'] = {
                    'engine': engine,
                    'df_forecast': df_forecast,
                    'alloc_df': alloc_df,
                    'df_target': df_target, # Tab 2, 3용
                    'risk_pivot_df': risk_pivot_df, # Tab 1 Risk용
                    'monthly_df': monthly_df,
                    'search_mode': search_mode
                }
                
        except Exception as e:
            st.error(f"시뮬레이션 중 오류 발생: {str(e)}")
            st.stop()

    # --- 결과 시각화 ---
    results = st.session_state['sim_results']
    if not results: st.stop()
    
    engine = results['engine']
    df_forecast = results['df_forecast']
    alloc_df = results['alloc_df']
    df_target = results['df_target']
    risk_pivot_df = results['risk_pivot_df']
    monthly_df = results['monthly_df']
    
    # 1. 제목
    selected_grades_str = ", ".join(sim_sel_grades) if sim_sel_grades else "전체 등급"
    st.subheader(f"📈 시뮬레이션 결과 ({selected_grades_str} / {search_mode})")

    fig_pred = go.Figure()
    
    # History - 그래프도 테이블과 동일한 범위 사용 (당월 제외, 작년 1월부터 시작)
    end_month_start = pd.to_datetime(end_date).replace(day=1)
    start_anchor = pd.Timestamp(end_month_start.year - 1, 1, 1)
    hist_df = df_target[(df_target['접수일자'] >= start_anchor) & (df_target['접수일자'] < end_month_start)]
    monthly_hist_series = hist_df.groupby(hist_df['접수일자'].dt.to_period('M')).size()
    monthly_hist_series.index = monthly_hist_series.index.to_timestamp()
    recent_hist = monthly_hist_series.tail(12)
    
    if not recent_hist.empty:
        last_date = recent_hist.index[-1]
        last_val = recent_hist.values[-1]
        
        # 미마감 당월이 있으면 실적과 연결 (예측 첫 달 x축과 동일하게 맞춤)
        if hasattr(engine, 'current_partial') and engine.current_partial > 0:
            current_month = df_forecast.index[0] if not df_forecast.empty else last_date + pd.DateOffset(months=1)
            
            # 실적 History + 당월 진행중을 하나의 선으로 (검은색 점선)
            combined_x = list(recent_hist.index) + [current_month]
            combined_y = list(recent_hist.values) + [engine.current_partial]
            
            fig_pred.add_trace(go.Scatter(
                x=combined_x, y=combined_y,
                mode='lines+markers',
                name='실적 (진행중 포함)',
                line=dict(color='black', width=2, dash='dot'),
                marker=dict(size=6)
            ))
            
            # 예측은 당월 마감(다음 달) 이후부터 시작 - 진행중과 연결하지 않음
            forecast_start_date = None  # 예측 첫 달부터 독립적으로 시작
        else:
            # 미마감 당월이 없으면 실적만 표시
            fig_pred.add_trace(go.Scatter(
                x=recent_hist.index, y=recent_hist.values,
                mode='lines+markers', name='실적 History',
                line=dict(color='black', width=2), marker=dict(size=6)
            ))
            forecast_start_date = None
    else:
        # 데이터가 없으면 monthly_df 기준
        last_date = monthly_df['ds'].iloc[-1] if not monthly_df.empty else pd.Timestamp.now()
        last_val = monthly_df['y'].iloc[-1] if not monthly_df.empty else 0
        forecast_start_date = None
        
    # Forecast - 예측 시작점부터 연결
    model_styles = {
        'Ensemble': {'color': '#6200ea', 'width': 5, 'dash': 'solid'},
        'AutoML':   {'color': '#e74c3c', 'width': 2, 'dash': 'dot'},
        'Prophet':  {'color': '#2ecc71', 'width': 1, 'dash': 'dot'},
        'SARIMAX':  {'color': '#3498db', 'width': 1, 'dash': 'dot'}
    }
    cols_sorted = [c for c in df_forecast.columns if c != 'Ensemble']
    if 'Ensemble' in df_forecast.columns: cols_sorted.append('Ensemble')
    
    for model_name in cols_sorted:
        style = model_styles.get(model_name, {'color': 'gray', 'width': 1, 'dash': 'dot'})
        mode_style = 'lines+markers' if model_name == 'Ensemble' else 'lines'
        
        # 예측은 독립적으로 표시 (당월 진행중과 연결하지 않음)
        fig_pred.add_trace(go.Scatter(
            x=df_forecast.index,
            y=df_forecast[model_name].values,
            mode=mode_style,
            name=f'{model_name} 예측',
            line=dict(color=style['color'], width=style['width'], dash=style['dash']),
            opacity=1.0 if model_name == 'Ensemble' else 0.5
        ))
        
    fig_pred.update_layout(
        height=500, 
        legend=dict(
            orientation="v", 
            yanchor="top", 
            y=0.99, 
            xanchor="right", 
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        hovermode="x unified", 
        xaxis=dict(showgrid=False), 
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0', rangemode='tozero'),
        plot_bgcolor='white'
    )
    
    # 예측 시작 시점 표시 (미마감 당월 마감 = 첫 예측 시점)
    if not df_forecast.empty:
        first_forecast = df_forecast.index[0]
        fig_pred.add_shape(
            type="line",
            x0=first_forecast, x1=first_forecast,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        fig_pred.add_annotation(
            x=first_forecast,
            y=1.0,
            yref="paper",
            text="예측 시작",
            showarrow=False,
            yshift=10,
            font=dict(color="red", size=10)
        )
    
    st.plotly_chart(fig_pred, width='stretch')
    
    if 'Ensemble' in df_forecast.columns:
        avg_pred = df_forecast['Ensemble'].mean()
        pred_period = f"{df_forecast.index[0].strftime('%Y-%m')} ~ {df_forecast.index[-1].strftime('%Y-%m')}"
        st.success(f"🏆 최종 앙상블(Ensemble) 예측 결과\n- 예측 기간: {pred_period}\n- 월평균: **{avg_pred:.0f}건**")
        
    st.divider()
    
    # [Risk 현황진단 - Tab 위에 표시]
    st.markdown("#### 🛡️ Risk 현황진단")
    if 'sim_final_view' in st.session_state and st.session_state['sim_final_view'] is not None:
        try:
            final_view = st.session_state['sim_final_view']
            risk_rows = final_view.reset_index()
            def _is_subtotal_row(row):
                val_minor = str(row.get('소분류', ''))
                if val_minor.startswith('소계_'): return True
                val_grade = str(row.get('등급기준', ''))
                if val_grade in ['합계']: return True
                return False
            
            risk_rows['__subtotal__'] = risk_rows.apply(_is_subtotal_row, axis=1)
            alerts_df = risk_rows[(risk_rows['🚨'].isin(['🔴','🟡'])) & (~risk_rows['__subtotal__'])]
            
            if not alerts_df.empty:
                c_left, c_right = st.columns(2)
                display_cols = ['등급기준', '대분류', '소분류', '진단']
                
                red_df = alerts_df[alerts_df['🚨'] == '🔴']
                yellow_df = alerts_df[alerts_df['🚨'] == '🟡']
                
                with c_left:
                    st.markdown(f"##### Red(🔴) 경보: {len(red_df)}건")
                    if not red_df.empty: 
                        st.dataframe(red_df[display_cols], width='stretch', hide_index=True)
                    else: 
                        st.info("레드 패턴 없음")
                with c_right:
                    st.markdown(f"##### Yellow(🟡) 주의: {len(yellow_df)}건")
                    if not yellow_df.empty: 
                        st.dataframe(yellow_df[display_cols], width='stretch', hide_index=True)
                    else: 
                        st.info("옐로우 패턴 없음")
            else:
                st.info("현재 경보 또는 주의 대상이 없습니다.")
        except Exception as e:
            st.warning(f"Risk 진단 오류: {e}")
    
    st.divider()
    
    # 3. 상세 분석 탭
    tab1, tab2, tab3 = st.tabs(["📊 통합 분석 테이블", "⏱️ Lag 분석", "📋 원본 데이터"])
    
    # [Tab 1] 통합 테이블 (Code B와 Risk 로직 동기화)
    with tab1:
        st.markdown("##### 📊 과거 실적 + 예측 시뮬레이션 통합 테이블")
        try:
            # 0. 전체 기간 패턴 계산 (예측 분배용)
            full_period_pivot = pd.DataFrame()
            if not df_target.empty:
                full_period_pivot = pd.pivot_table(
                    df_target,
                    index=['등급기준', '대분류', '소분류'],
                    values='건수', aggfunc='sum', fill_value=0
                )
            
            # 1. Base History Pivot (표시용 - 작년 1월부터 당월 직전까지)
            end_month_start = pd.to_datetime(end_date).replace(day=1)
            start_anchor = pd.Timestamp(end_month_start.year - 1, 1, 1)
            historical_12m = df_target[(df_target['접수일자'] >= start_anchor) & (df_target['접수일자'] < end_month_start)].copy()
            
            historical_12m['월'] = historical_12m['접수일자'].dt.strftime('%Y-%m')
            
            pivot_hist = pd.DataFrame()
            if not historical_12m.empty:
                pivot_hist = pd.pivot_table(
                    historical_12m,
                    index=['등급기준', '대분류', '소분류'],
                    columns='월',
                    values='건수', aggfunc='sum', fill_value=0
                )
            
            # 1.5. Forecast Data Pivot - 시간 가중 패턴 기반 예측 분배 (당월도 예측값으로 포함)
            pivot_pred = pd.DataFrame()
            pivot_curr_actual = pd.DataFrame()
            pivot_curr_pred = pd.DataFrame()
            if not alloc_df.empty and not df_target.empty:
                # === 시간 가중 비율 계산 (최근 데이터 중시, 당월 제외) ===
                # 월별 데이터 생성 (당월 제외)
                df_monthly = df_target.copy()
                df_monthly['접수월'] = df_monthly['접수일자'].dt.to_period('M')
                
                # 기간 제한: 작년 1월부터 당월 직전까지
                end_month = pd.to_datetime(end_date).to_period('M')
                start_anchor = pd.Timestamp(end_month.year - 1, 1, 1)
                df_monthly = df_monthly[
                    (df_monthly['접수일자'] >= start_anchor) &
                    (df_monthly['접수월'] != end_month)
                ]
                
                df_monthly['년월'] = df_monthly['접수일자'].dt.to_period('M')
                
                # 등급기준|대분류|소분류 × 년월 피벗
                monthly_pivot = pd.pivot_table(
                    df_monthly,
                    index=['등급기준', '대분류', '소분류'],
                    columns='년월',
                    values='건수',
                    aggfunc='sum',
                    fill_value=0
                )
                
                if not monthly_pivot.empty:
                    # 시간 가중치 계산 (지수 감쇠: 최근일수록 높은 가중치)
                    n_months = len(monthly_pivot.columns)
                    # decay_rate: 0.9 = 최근 12개월에 높은 가중치, 과거는 급격히 감소
                    decay_rate = 0.92
                    time_weights = np.array([decay_rate ** (n_months - i - 1) for i in range(n_months)])
                    time_weights = time_weights / time_weights.sum()  # 정규화
                    
                    # 각 조합별 가중 평균 계산
                    weighted_totals = (monthly_pivot.values * time_weights).sum(axis=1)
                    weighted_series = pd.Series(weighted_totals, index=monthly_pivot.index)
                    
                    # 소멸 추세 감지 및 비율 조정
                    recent_12m = monthly_pivot.iloc[:, -12:] if monthly_pivot.shape[1] >= 12 else monthly_pivot
                    recent_avg = recent_12m.mean(axis=1)
                    historical_avg = monthly_pivot.mean(axis=1)
                    
                    # 소멸 비율 계산 (최근 평균 / 전체 평균)
                    extinction_ratio = recent_avg / historical_avg.replace(0, 1)
                    
                    # 소멸 추세 감지 (최근 평균이 전체 평균의 20% 이하)
                    is_extinct = extinction_ratio < 0.2
                    
                    # 가중치 조정: 소멸 추세면 최근 데이터만 사용
                    final_ratios = weighted_series.copy()
                    for idx in weighted_series.index:
                        if is_extinct.loc[idx]:
                            # 소멸 추세: 최근 6개월만 사용
                            recent_6m = recent_12m.loc[idx].tail(6)
                            final_ratios.loc[idx] = recent_6m.mean()
                    
                    # 전체 합으로 정규화하여 비율 계산
                    total_sum = final_ratios.sum()
                    if total_sum > 0:
                        hist_ratio = final_ratios / total_sum
                    else:
                        hist_ratio = pd.Series(0, index=final_ratios.index)
                    
                    # 예측 총량 추출 (Ensemble 기준)
                    if 'Ensemble' in df_forecast.columns:
                        forecast_totals = df_forecast['Ensemble']
                    else:
                        forecast_totals = df_forecast.mean(axis=1)
                    
                    # 예측 월별 컬럼 생성
                    pred_data = []
                    for idx in hist_ratio.index:
                        ratio = hist_ratio.loc[idx]
                        if ratio > 0:  # 비율이 0보다 큰 경우만 예측 생성
                            for month_idx, month_dt in enumerate(df_forecast.index):
                                month_str = month_dt.strftime('%Y-%m')
                                pred_val = forecast_totals.iloc[month_idx] * ratio
                                
                                pred_data.append({
                                    '등급기준': idx[0],
                                    '대분류': idx[1],
                                    '소분류': idx[2],
                                    '예측월': month_str,
                                    '예측건수': pred_val
                                })
                    
                    if pred_data:
                        pred_df = pd.DataFrame(pred_data)
                        pivot_pred = pd.pivot_table(
                            pred_df,
                            index=['등급기준', '대분류', '소분류'],
                            columns='예측월',
                            values='예측건수',
                            aggfunc='sum',
                            fill_value=0
                        )

            # Current month actual/forecast columns
            current_month = pd.to_datetime(end_date).to_period('M')
            current_month_str = current_month.strftime('%Y-%m')
            current_actual_col = f"{current_month_str}(실제)"
            current_pred_col = f"{current_month_str}(예측)"

            # Actual current month pivot
            current_month_df = df_target[df_target['접수일자'].dt.to_period('M') == current_month]
            if not current_month_df.empty:
                pivot_curr_actual = pd.pivot_table(
                    current_month_df,
                    index=['등급기준', '대분류', '소분류'],
                    values='건수', aggfunc='sum', fill_value=0
                )
                pivot_curr_actual.columns = [current_actual_col]

            # Forecast current month pivot (renamed) and remove original column to avoid duplicates
            if not pivot_pred.empty and current_month_str in pivot_pred.columns:
                pivot_curr_pred = pivot_pred[[current_month_str]].rename(columns={current_month_str: current_pred_col})
                pivot_pred = pivot_pred.drop(columns=[current_month_str])
            
            # 2. Risk Scoring (Forecast 기반 예측 진단)
            risk_data = []
            target_month_str = pd.to_datetime(end_date).strftime('%Y-%m')
            
            target_indices = pivot_hist.index if not pivot_hist.empty else []

            for idx in target_indices:
                grade, major, minor = idx
                try:
                    # 과거 데이터 추출 (최근 12개월)
                    if idx in pivot_hist.index:
                        hist_series = pivot_hist.loc[idx]
                    else:
                        hist_series = pd.Series(0, index=pivot_hist.columns)
                    
                    # 현재 값 (당월 실적 = 과거 데이터의 마지막 값)
                    current_actual = hist_series.iloc[-1] if len(hist_series) > 0 else 0
                    
                    # 당월 예측값 추출
                    current_forecast = 0
                    if idx in pivot_curr_pred.index:
                        current_forecast = pivot_curr_pred.loc[idx, current_pred_col] if current_pred_col in pivot_curr_pred.columns else 0
                    
                    # 미래 예측 데이터 추출
                    if idx in pivot_pred.index:
                        fcst_series = pivot_pred.loc[idx]
                    else:
                        fcst_series = pd.Series(0, index=pivot_pred.columns)
                    
                    # ForecastRiskAnalyzer 호출
                    analyzer = ForecastRiskAnalyzer(
                        historical_series=hist_series,
                        current_actual=current_actual,
                        current_forecast=current_forecast,
                        forecast_series=fcst_series,
                        grade=grade
                    )
                    result = analyzer.analyze()
                    
                    sig = result['status']
                    reason = result['insight']
                    
                except Exception as e:
                    sig, reason = ('⚪', f"분석오류: {str(e)}")
                
                risk_data.append({'등급기준': grade, '대분류': major, '소분류': minor, '🚨': sig, '진단': reason})
            
            df_risk = pd.DataFrame(risk_data)
            if not df_risk.empty:
                df_risk = df_risk.set_index(['등급기준', '대분류', '소분류'])
            
            # 3. Final Merge
            dfs = []
            if not df_risk.empty: dfs.append(df_risk)
            if not pivot_hist.empty: dfs.append(pivot_hist)
            if not pivot_curr_actual.empty: dfs.append(pivot_curr_actual)
            if not pivot_curr_pred.empty: dfs.append(pivot_curr_pred)
            if not pivot_pred.empty: dfs.append(pivot_pred)
            
            if dfs:
                final_view = pd.concat(dfs, axis=1).fillna(0)
                
                # Column Sort (meta first, then past year → past-year avg → current actual/pred → future fcst → fcst avg)
                meta = ['🚨', '진단']
                hist_cols_order = [c for c in sorted(pivot_hist.columns) if c in final_view.columns]
                curr_cols = [c for c in [current_actual_col, current_pred_col] if c in final_view.columns]
                pred_cols_order = [c for c in sorted(pivot_pred.columns) if c in final_view.columns]
                ordered = hist_cols_order + curr_cols + pred_cols_order
                # keep any remaining columns (e.g., averages to be appended later)
                ordered += [c for c in final_view.columns if c not in meta and c not in ordered]
                final_view = final_view[meta + ordered]
                
                # Value Restoration
                final_view['🚨'] = final_view['🚨'].replace(0, '⚪')
                final_view['진단'] = final_view['진단'].replace(0, '-')

                # 평균 컬럼 추가 (과거연도 라벨은 실제 연도로 표시)
                hist_cols = [c for c in pivot_hist.columns if c in final_view.columns]
                pred_cols = [c for c in pivot_pred.columns if c in final_view.columns]
                hist_base_year = pd.Timestamp(pd.to_datetime(end_date).year - 1, 1, 1).year
                hist_avg_col = f"{hist_base_year}년 월평균"
                if hist_cols:
                    final_view[hist_avg_col] = final_view[hist_cols].mean(axis=1)
                if pred_cols:
                    final_view['예측 월평균'] = final_view[pred_cols].mean(axis=1)

                # Column re-order after adding averages: past year → past-year avg → current actual/pred → future fcst → fcst avg
                hist_cols_order = [c for c in sorted(pivot_hist.columns) if c in final_view.columns]
                curr_cols = [c for c in [current_actual_col, current_pred_col] if c in final_view.columns]
                pred_cols_order = [c for c in sorted(pivot_pred.columns) if c in final_view.columns]
                ordered = []
                ordered += hist_cols_order
                if hist_avg_col in final_view.columns: ordered.append(hist_avg_col)
                ordered += curr_cols
                ordered += [c for c in pred_cols_order if c not in ordered]
                if '예측 월평균' in final_view.columns: ordered.append('예측 월평균')
                # include any remaining columns (fallback)
                ordered += [c for c in final_view.columns if c not in meta and c not in ordered]
                final_view = final_view[meta + ordered]
                
                # 소계 및 합계 추가
                numeric_cols = final_view.select_dtypes(include=[np.number]).columns
                
                # 등급기준|대분류별 소계 삽입
                result_data = []
                subtotal_counter = 0
                
                for (grade, major), group in final_view.groupby(level=[0, 1]):
                    # 그룹 데이터 추가
                    for idx, row in group.iterrows():
                        result_data.append({
                            '등급기준': idx[0],
                            '대분류': idx[1],
                            '소분류': idx[2],
                            **row.to_dict()
                        })
                    
                    # 소계 행 생성 (axis=0으로 행을 합산, numeric_only로 숫자만 처리)
                    subtotal_vals = group[numeric_cols].sum(axis=0, numeric_only=True)
                    subtotal_dict = {
                        '등급기준': grade,
                        '대분류': major,
                        '소분류': f'소계_{subtotal_counter}',  # 고유 인덱스
                        '🚨': '',
                        '진단': ''
                    }
                    for col in numeric_cols:
                        subtotal_dict[col] = subtotal_vals.get(col, 0)
                    result_data.append(subtotal_dict)
                    subtotal_counter += 1
                
                # 재구성
                final_view = pd.DataFrame(result_data)
                final_view = final_view.set_index(['등급기준', '대분류', '소분류'])
                
                # 전체 합계 추가 (소계 행 제외하고 실제 데이터 행만 계산)
                mask_data = ~final_view.index.get_level_values(2).astype(str).str.startswith('소계_')
                total_vals = final_view.loc[mask_data, numeric_cols].sum(axis=0, numeric_only=True)
                total_dict = {
                    '등급기준': '합계',
                    '대분류': '-',
                    '소분류': '-',
                    '🚨': '-',
                    '진단': '-'
                }
                for col in numeric_cols:
                    total_dict[col] = total_vals.get(col, 0)
                
                # 합계를 DataFrame으로 추가
                total_df = pd.DataFrame([total_dict]).set_index(['등급기준', '대분류', '소분류'])
                final_view = pd.concat([final_view, total_df])
                
                # 당월 열 및 예측 컬럼 식별
                target_month_str = current_actual_col
                pred_cols = [c for c in pivot_pred.columns if c in final_view.columns]
                if current_pred_col in final_view.columns:
                    pred_cols.append(current_pred_col)
                
                # Styling
                def highlight_risk(row):
                    styles = []
                    hist_avg = row.get(hist_avg_col, np.nan)
                    
                    # 예측 열 중 상위 2개월만 히트맵 적용
                    top_2_pred_cols = []
                    if len(pred_cols) >= 2:
                        # 해당 행의 예측값 추출 및 내림차순 정렬
                        pred_vals = {col: row.get(col, 0) for col in pred_cols}
                        top_2_pred_cols = sorted(pred_vals.items(), key=lambda x: x[1], reverse=True)[:2]
                        top_2_pred_cols = [col for col, val in top_2_pred_cols]
                    elif len(pred_cols) == 1:
                        top_2_pred_cols = pred_cols

                    for col in final_view.columns:
                        style_parts = []
                        if isinstance(row.name, tuple) and len(row.name) >= 3:
                            if str(row.name[2]).startswith('소계_'):
                                style_parts.append('background-color: #e5e7eb; font-weight: bold; font-size: 0.9em')
                            elif row.name[0] == '합계':
                                style_parts.append('background-color: #f3f4f6; font-weight: bold')
                        sig = row.get('🚨', '')
                        if sig == '🔴':
                            style_parts.append('background-color: #fee2e2; color: #991b1b')
                        elif sig == '🟡':
                            style_parts.append('background-color: #fef3c7; color: #92400e')
                        if col == target_month_str:
                            style_parts.append('color: darkred; font-weight: bold')
                        # 작년 월 평균 열에 회색 배경
                        if col == hist_avg_col:
                            style_parts.append('background-color: #f3f4f6; font-weight: bold')
                        # 예측 열 중 상위 2개월만 히트맵 (rgba 점진적 색칠)
                        if col in top_2_pred_cols and pd.notnull(hist_avg):
                            cell_val = row.get(col, np.nan)
                            if pd.notnull(cell_val):
                                diff = cell_val - hist_avg
                                if diff > 0 and hist_avg > 0:
                                    ratio = min(1.0, diff / hist_avg)
                                    alpha = 0.3 + 0.3 * ratio  # 0.3~0.6 사이 투명도
                                    style_parts.append(f'background-color: rgba(248, 113, 113, {alpha:.2f}); color: #7f1d1d')
                        styles.append('; '.join(style_parts))
                    return styles
                
                # 포맷팅할 컬럼만 필터링 (실제 숫자 타입만)
                format_dict = {}
                for c in numeric_cols:
                    if c in final_view.columns and pd.api.types.is_numeric_dtype(final_view[c]):
                        format_dict[c] = "{:,.1f}"
                
                st.dataframe(
                    final_view.style.format(format_dict).apply(highlight_risk, axis=1),
                    width='stretch', height=500
                )
                
                csv = final_view.to_csv().encode('utf-8-sig')
                st.download_button("📥 통합 분석 결과 다운로드 (CSV)", csv, f"Simulation_Result.csv", "text/csv")
                
                # final_view를 session_state에 저장 (Tab 외부에서 사용)
                st.session_state['sim_final_view'] = final_view
                st.session_state['sim_hist_avg_col'] = hist_avg_col
                
            else:
                st.info("표시할 데이터가 없습니다.")
                
        except Exception as e:
            st.error(f"테이블 생성 오류: {str(e)}")
            
    # [Tab 2] Lag 분석 (Code B와 완전 동일)
    with tab2:
        st.markdown("##### ⏱️ Lag 분석 (제조 ~ 접수 소요기간)")
        lag_stats = calculate_lag_stats(df_target)
        
        if lag_stats and lag_stats['count'] > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("평균 Lag", f"{lag_stats['mean']:.1f} 일")
            c2.metric("중앙값 Lag", f"{lag_stats['p50']:.1f} 일")
            c3.metric("대상 건수", f"{lag_stats['count']:,} 건")
            
            if 'Lag_Valid' in df_target.columns and 'Lag_Days' in df_target.columns:
                valid_lag_df = df_target[df_target['Lag_Valid'] == True]
                fig = px.histogram(valid_lag_df, x='Lag_Days', nbins=50, title="Lag Days Distribution")
                st.plotly_chart(fig, width='stretch')
        else:
            st.warning("유효 데이터 없음")
            
    # [Tab 3] 원본 데이터 (Code B와 완전 동일)
    with tab3:
        st.markdown("##### 📋 원본 데이터 (필터링 적용)")
        st.dataframe(df_target, width='stretch', height=500)
        st.caption(f"※ 조회된 데이터 건수: {len(df_target):,}건")

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
