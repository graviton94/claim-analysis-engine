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
        
        # PyArrow Filter Expression (속도 최적화)
        # 플랜트와 대분류 조건으로 파티션을 필터링하여 읽음
        filter_expr = (ds.field('플랜트') == plant) & (ds.field('대분류') == major)
        
        # 컬럼 제한 없이 모든 컬럼 읽기
        table = dataset.to_table(filter=filter_expr)
        df_full = table.to_pandas()
        
        if '접수일자' in df_full.columns:
            df_full['접수일자'] = pd.to_datetime(df_full['접수일자'])
            
        # 모드에 따른 필터링 적용 (실적 모드일 경우)
        if mode == "실적 (Performance)":
            if '불만원인' in df_full.columns:
                reasons = ['고객불만족', '구매불만', '제조불만']
                df_full = df_full[df_full['불만원인'].isin(reasons)]
            
            if '사업부문' in df_full.columns:
                biz_units = ['식품', 'B2B식품']
                df_full = df_full[df_full['사업부문'].isin(biz_units)]
                
        return df_full
    except Exception as e:
        st.error(f"백데이터 로드 실패: {e}")
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
    
    col_l, col_r = st.columns([1.2, 1])
    
    # --- 1. Top-down Forecasting ---
    with col_l:
        st.subheader("🏁 모델 경합 (Model Competition)")
        
        # 그래프 시각화
        fig_pred = go.Figure()
        
        # 1. 과거 데이터
        recent_hist = engine.train_data.tail(12)
        last_date = recent_hist.index[-1]
        last_val = recent_hist.values[-1]
        
        fig_pred.add_trace(go.Scatter(
            x=recent_hist.index, y=recent_hist.values,
            mode='lines+markers', name='실적 History',
            line=dict(color='black', width=2),
            marker=dict(size=6)
        ))
        
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
            title=dict(text=f"<b>향후 {forecast_months}개월 예측 시나리오</b>", font=dict(size=20)),
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            plot_bgcolor='white'
        )
        fig_pred.add_vline(x=last_date, line_width=1, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # 가중치 시각화
        st.divider()
        st.markdown("#### ⚖️ 모델별 가중치 (Dynamic Weights)")
        weights = results['model_weights']
        if weights:
            cols = st.columns(len(weights))
            for i, (model, w) in enumerate(weights.items()):
                cols[i].metric(label=model, value=f"{w*100:.1f}%")

        if 'Ensemble' in df_forecast.columns:
            avg_pred = df_forecast['Ensemble'].mean()
            st.success(f"🏆 최종 앙상블(Ensemble) 예측 결과, 향후 월평균 **{avg_pred:.0f}건**이 예상됩니다.")

    # --- 2. Bottom-up Allocation (Seasonal) ---
    with col_r:
        st.subheader("🧩 소분류 배분 (Seasonal Allocation)")
        st.caption("앙상블(Ensemble) 예측 총량을 과거 동월 비중(Ratio)에 따라 하위 소분류로 배분합니다.")
        
        if not alloc_df.empty:
            pivot_alloc = alloc_df.pivot_table(
                index='소분류', 
                columns='예측월', 
                values='예측건수', 
                aggfunc='sum',
                fill_value=0
            )
            pivot_alloc = pivot_alloc.fillna(0)
            pivot_alloc.loc['Total'] = pivot_alloc.sum()
            
            # 스타일링
            st.dataframe(
                pivot_alloc.style
                    .format("{:,.1f}")
                    .background_gradient(
                        cmap="Reds", 
                        subset=(pivot_alloc.index[:-1], pivot_alloc.columns)
                    )
                    .apply(lambda x: ['font-weight: bold' if x.name == 'Total' else '' for _ in x], axis=1),
                use_container_width=True,
                height=400
            )
            
            st.divider()
            b1, b2 = st.columns(2)
            
            # (1) 배분 결과 CSV
            csv_alloc = alloc_df.to_csv(index=False).encode('utf-8-sig')
            major_str = "_".join(selected_categories[:2]) if selected_categories else "All"
            b1.download_button(
                label="📥 배분 결과 (CSV)",
                data=csv_alloc,
                file_name=f"Allocation_{sel_plant}_{major_str}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # (2) 백데이터(전체 컬럼) CSV
            if not df_full_backdata.empty:
                csv_raw = df_full_backdata.to_csv(index=False).encode('utf-8-sig')
                b2.download_button(
                    label="💾 원본 백데이터 (CSV)",
                    data=csv_raw,
                    file_name=f"BackData_Full_{sel_plant}_{major_str}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                b2.warning("백데이터 로드 실패")
                
        else:
            st.warning("배분할 하위 데이터가 부족하거나 예측 결과(Ensemble)가 없습니다.")

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