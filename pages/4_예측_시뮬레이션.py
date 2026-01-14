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
from core.analytics import calculate_advanced_risk_score, calculate_lag_stats, prepare_risk_data

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
    
    # 쿼리 파라미터로 설정된 기본값 또는 전체 선택
    if 'sim_step3_grades' not in st.session_state:
        st.session_state['sim_step3_grades'] = grade_options
    
    # 다중선택에서 쿼리 파라미터의 등급만 기본으로 선택되도록
    default_grades = st.session_state.get('sim_step3_grades', grade_options)
    sim_sel_grades = st.multiselect(
        "분석할 등급을 선택하세요:", 
        grade_options, 
        default=default_grades,
        key='sim_step3_grades'
    )
    
    # 2. 대분류 필터
    st.markdown("")
    temp_df = df_for_options
    if sim_sel_grades:
        temp_df = temp_df[temp_df['등급기준'].isin(sim_sel_grades)]
        
    category_options = sorted(temp_df['대분류'].dropna().unique())
    
    if 'sim_step3_categories' not in st.session_state:
        st.session_state['sim_step3_categories'] = category_options
    
    # 다중선택에서 쿼리 파라미터의 대분류만 기본으로 선택되도록
    default_categories = st.session_state.get('sim_step3_categories', category_options)
    # 필터링된 카테고리 중에 기본값이 있는지 확인
    default_categories = [cat for cat in default_categories if cat in category_options]
    if not default_categories:  # 기본값이 필터링되었으면 전체
        default_categories = category_options
    
    sim_sel_categories = st.multiselect(
        "분석할 대분류를 선택하세요:", 
        category_options, 
        default=default_categories,
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
btn_run = st.button("🚀 시뮬레이션 시작", type="primary", use_container_width=True, disabled=not can_run, on_click=run_simulation)

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
    
    # History
    recent_hist = engine.train_data.tail(12)
    last_date = recent_hist.index[-1] if not recent_hist.empty else monthly_df['ds'].iloc[-1]
    last_val = recent_hist.values[-1] if not recent_hist.empty else monthly_df['y'].iloc[-1]
    
    if not recent_hist.empty:
        fig_pred.add_trace(go.Scatter(
            x=recent_hist.index, y=recent_hist.values,
            mode='lines+markers', name='실적 History',
            line=dict(color='black', width=2), marker=dict(size=6)
        ))
        
    # Forecast
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
        pred_x = [last_date] + list(df_forecast.index)
        pred_y = [last_val] + list(df_forecast[model_name].values)
        mode_style = 'lines+markers' if model_name == 'Ensemble' else 'lines'
        
        fig_pred.add_trace(go.Scatter(
            x=pred_x, y=pred_y, mode=mode_style, name=f'{model_name} 예측',
            line=dict(color=style['color'], width=style['width'], dash=style['dash']),
            opacity=1.0 if model_name == 'Ensemble' else 0.5
        ))
        
    fig_pred.update_layout(
        height=500, legend=dict(orientation="h", y=1.02, x=1),
        hovermode="x unified", xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
        plot_bgcolor='white'
    )
    fig_pred.add_vline(x=last_date, line_width=1, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_pred, use_container_width=True)
    
    if 'Ensemble' in df_forecast.columns:
        avg_pred = df_forecast['Ensemble'].mean()
        st.success(f"🏆 최종 앙상블(Ensemble) 예측 결과, 향후 월평균 **{avg_pred:.0f}건**이 예상됩니다.")
        
    st.divider()
    
    # 3. 상세 분석 탭
    tab1, tab2, tab3 = st.tabs(["📊 통합 분석 테이블", "⏱️ Lag 분석", "📋 원본 데이터"])
    
    # [Tab 1] 통합 테이블 (Code B와 Risk 로직 동기화)
    with tab1:
        st.markdown("##### 📊 과거 실적 + 예측 시뮬레이션 통합 테이블")
        try:
            # 1. Base History Pivot
            cutoff_date = pd.to_datetime(end_date) - relativedelta(months=12)
            historical_12m = df_target[df_target['접수일자'] >= cutoff_date].copy()
            historical_12m['월'] = historical_12m['접수일자'].dt.strftime('%Y-%m')
            
            pivot_hist = pd.DataFrame()
            if not historical_12m.empty:
                pivot_hist = pd.pivot_table(
                    historical_12m,
                    index=['등급기준', '대분류', '소분류'],
                    columns='월',
                    values='건수', aggfunc='sum', fill_value=0
                )
            
            # 2. Risk Scoring (Code B Logic: Zero-Filled Data)
            # risk_pivot_df는 이미 Zero-filled & MultiIndex 상태임
            risk_data = []
            target_month_str = pd.to_datetime(end_date).strftime('%Y-%m')
            
            # loop over base pivot indices (or alloc_mapped indices if needed)
            # 여기서는 pivot_hist의 인덱스를 기준으로 함
            # (만약 alloc_df에만 있는 신규 항목이 있다면 추가 로직 필요하지만, 보통 과거 이력 기반임)
            
            target_indices = pivot_hist.index if not pivot_hist.empty else []
            if not pivot_hist.empty and not alloc_df.empty:
                # Merge indices from history and forecast
                # (생략: 복잡도 증가 방지, History 있는 항목 위주)
                pass

            for idx in target_indices:
                grade, major, minor = idx
                try:
                    # Risk Pivot에서 데이터 추출 (없으면 0)
                    if idx in risk_pivot_df.index:
                        series_data = risk_pivot_df.loc[idx]
                    else:
                        series_data = pd.Series(0, index=risk_pivot_df.columns)
                    
                    # Risk Engine 호출
                    sig, score, reason = calculate_advanced_risk_score(series_data, target_month_str, grade=grade)
                except Exception as e:
                    sig, score, reason = ('⚪', 0, f"Err: {str(e)}")
                
                risk_data.append({'등급기준': grade, '대분류': major, '소분류': minor, '🚨': sig, '진단': reason})
            
            df_risk = pd.DataFrame(risk_data)
            if not df_risk.empty:
                df_risk = df_risk.set_index(['등급기준', '대분류', '소분류'])
            
            # 3. Forecast Data Pivot
            pivot_pred = pd.DataFrame()
            if not alloc_df.empty:
                mapping = df_target[['소분류', '등급기준']].drop_duplicates().set_index('소분류')
                alloc_mapped = alloc_df.copy()
                # 등급기준이 누락되었을 수 있으므로 매핑
                alloc_mapped['등급기준'] = alloc_mapped['소분류'].map(mapping['등급기준']).fillna('미지정')
                
                pivot_pred = pd.pivot_table(
                    alloc_mapped,
                    index=['등급기준', '대분류', '소분류'],
                    columns='예측월',
                    values='예측건수', aggfunc='sum', fill_value=0
                )
            
            # 4. Final Merge
            dfs = []
            if not df_risk.empty: dfs.append(df_risk)
            if not pivot_hist.empty: dfs.append(pivot_hist)
            if not pivot_pred.empty: dfs.append(pivot_pred)
            
            if dfs:
                final_view = pd.concat(dfs, axis=1).fillna(0)
                
                # Column Sort
                meta = ['🚨', '진단']
                others = sorted([c for c in final_view.columns if c not in meta])
                final_view = final_view[meta + others]
                
                # Value Restoration
                final_view['🚨'] = final_view['🚨'].replace(0, '⚪')
                final_view['진단'] = final_view['진단'].replace(0, '-')
                
                # Total Row
                numeric_cols = final_view.select_dtypes(include=[np.number]).columns
                total_vals = final_view[numeric_cols].sum()
                total_idx = ('합계', '-', '-')
                final_view.loc[total_idx, numeric_cols] = total_vals
                final_view.loc[total_idx, ['🚨', '진단']] = ['-', '-']
                
                # Styling
                def highlight_risk(row):
                    if row.name == total_idx:
                        return ['background-color: #f3f4f6; font-weight: bold'] * len(row)
                    sig = row['🚨']
                    if sig == '🔴': return ['background-color: #fee2e2; color: #991b1b'] * len(row)
                    elif sig == '🟡': return ['background-color: #fef3c7; color: #92400e'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    final_view.style.format({c: "{:,.1f}" for c in numeric_cols}).apply(highlight_risk, axis=1),
                    use_container_width=True, height=500
                )
                
                csv = final_view.to_csv().encode('utf-8-sig')
                st.download_button("📥 통합 분석 결과 다운로드 (CSV)", csv, f"Simulation_Result.csv", "text/csv")
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
        st.dataframe(df_target, use_container_width=True, height=500)
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
