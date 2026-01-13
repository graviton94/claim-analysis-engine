import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
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
        cols = ['플랜트', '대분류', '소분류', '접수일자', '건수', '불만원인', '사업부문']
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
# 3. 사이드바: 시뮬레이션 설정
# ==============================================================================
st.sidebar.header("1. 분석 대상 설정")

# 1-0. 분석 모드
mode_select = st.sidebar.radio(
    "📊 조회 모드",
    ["인입 (Inflow)", "실적 (Performance)"],
    help="실적 모드는 '고객불만족/구매불만/제조불만' 및 '식품/B2B식품' 건만 필터링합니다."
)

# 데이터 필터링 로직 (UI용)
if mode_select == "실적 (Performance)":
    if '불만원인' in df_raw.columns:
        reasons = ['고객불만족', '구매불만', '제조불만']
        df_filtered = df_raw[df_raw['불만원인'].isin(reasons)].copy()
    else:
        df_filtered = df_raw.copy()
        
    if '사업부문' in df_filtered.columns:
        biz_units = ['식품', 'B2B식품']
        df_filtered = df_filtered[df_filtered['사업부문'].isin(biz_units)]
else:
    df_filtered = df_raw.copy()

# Step 1: Plant
plants = sorted(df_filtered['플랜트'].dropna().unique()) if not df_filtered.empty else []
sel_plant = st.sidebar.selectbox("🏭 플랜트 선택", plants)

# Step 2: Major Category
if sel_plant:
    df_plant = df_filtered[df_filtered['플랜트'] == sel_plant]
    majors = sorted(df_plant['대분류'].dropna().unique())
    sel_major = st.sidebar.selectbox("📂 대분류 선택", majors)
else:
    df_plant = pd.DataFrame()
    majors = []
    sel_major = None

# Step 3: Final Target Data (UI Analysis)
if sel_major:
    df_target = df_plant[df_plant['대분류'] == sel_major].copy()
    
    st.sidebar.divider()
    st.sidebar.info(f"📊 분석 대상: {len(df_target):,}건")
    if not df_target.empty:
        valid_dates = df_target['접수일자'].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            st.sidebar.caption(f"기간: {min_date} ~ {max_date}")
else:
    df_target = pd.DataFrame()

# Step 4: Simulation Params
st.sidebar.header("2. 실험 파라미터")
forecast_months = st.sidebar.slider("예측 기간 (개월)", 3, 12, 6)
n_trials = st.sidebar.number_input("AutoML 시도 횟수", 10, 50, 15, help="높을수록 정확하지만 느려집니다.")

# Action Button
# [FIX] Session State 초기화
if 'sim_results' not in st.session_state:
    st.session_state['sim_results'] = None

def run_simulation():
    st.session_state['run_clicked'] = True
    # 이전 결과 초기화
    st.session_state['sim_results'] = None

btn_run = st.sidebar.button("🚀 시뮬레이션 시작", type="primary", use_container_width=True, disabled=df_target.empty, on_click=run_simulation)

# ==============================================================================
# 4. 메인 화면: 분석 및 시각화
# ==============================================================================

if df_target.empty:
    st.warning("데이터를 선택해주세요 (필터 조건에 맞는 데이터가 없을 수 있습니다).")
else:
    # [A] 데이터 미리보기 (Expandable)
    with st.expander("🔍 선택한 대분류의 과거 이력 보기", expanded=True):
        daily_trend = df_target.set_index('접수일자').resample('M')['건수'].sum()
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=daily_trend.index, y=daily_trend.values,
            mode='lines+markers', name='실적',
            line=dict(color='black', width=2)
        ))
        fig_hist.update_layout(
            title=f"{sel_plant} > {sel_major} 월별 추이 ({mode_select})",
            height=300, margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title=None, yaxis_title="건수"
        )
        st.plotly_chart(fig_hist, width='stretch')

# [B] 시뮬레이션 실행 및 결과 표시 (Session State 기반)
# 버튼을 눌렀거나, 이미 결과가 저장되어 있는 경우 실행
if (st.session_state.get('run_clicked') or st.session_state.get('sim_results')) and not df_target.empty:
    st.divider()
    
    # --- 계산 로직 (결과가 없을 때만 실행) ---
    if st.session_state['sim_results'] is None:
        try:
            with st.spinner("🧠 AI 모델 엔진 가동 중... (예측 및 백데이터 로딩)"):
                # 1. 예측 엔진 실행
                if '건수' not in df_target.columns: df_target['건수'] = 1
                engine = SimulationEngine(df_target, date_col='접수일자', val_col='건수')
                df_forecast = engine.run_competition(periods=forecast_months)
                
                # 2. 배분 로직 실행
                alloc_df = engine.predict_with_allocation(
                    plant=sel_plant,
                    major_category=sel_major,
                    sub_df=df_target,
                    periods=forecast_months,
                    forecast_df=df_forecast
                )

                # 3. [NEW] 다운로드용 전체 컬럼 데이터 로드 (Heavy Task)
                df_full_backdata = load_full_target_data(sel_plant, sel_major, mode_select)
                
                # 4. 결과 저장
                st.session_state['sim_results'] = {
                    'engine': engine,
                    'df_forecast': df_forecast,
                    'alloc_df': alloc_df,
                    'df_full_backdata': df_full_backdata,
                    'model_weights': engine.model_weights
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
            
            # 스타일링 (width=None 이슈 해결됨)
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
            b1.download_button(
                label="📥 배분 결과 (CSV)",
                data=csv_alloc,
                file_name=f"Allocation_{sel_plant}_{sel_major}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # (2) 백데이터(전체 컬럼) CSV
            # [FIX] Session State에 저장된 Full Data 사용
            if not df_full_backdata.empty:
                csv_raw = df_full_backdata.to_csv(index=False).encode('utf-8-sig')
                b2.download_button(
                    label="💾 원본 백데이터 (CSV)",
                    data=csv_raw,
                    file_name=f"BackData_Full_{sel_plant}_{sel_major}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                b2.warning("백데이터 로드 실패")
                
        else:
            st.warning("배분할 하위 데이터가 부족하거나 예측 결과(Ensemble)가 없습니다.")

elif not btn_run and not st.session_state.get('run_clicked'):
    st.info("👈 왼쪽 사이드바에서 대상을 선택하고 [시뮬레이션 시작] 버튼을 눌러주세요.")
    
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