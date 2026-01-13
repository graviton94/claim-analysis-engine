import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pyarrow.dataset as ds
import os
from pathlib import Path
from dateutil.relativedelta import relativedelta
import numpy as np
import base64
from io import BytesIO

# [Core Engine] Phase 2.8 + 2.5(Refactor)
from core.storage import DATA_HUB_PATH, load_and_filter_data, get_claim_keys
from core.analytics import calculate_advanced_risk_score, prepare_risk_data
from core.forecasting import ForecastEngine

# --- [CONFIG] Color System ---
COLOR_RED = "#EF151E"
COLOR_YELLOW = "#FF9700"
COLOR_BLUE = "#006ECD"
COLOR_GRAY = "#9ca3af"

# --- 0. Helper Functions (UI & Data) ---

def format_diagnosis(diagnosis_str):
    """진단 결과를 category와 detail로 분리해서 표시"""
    if not diagnosis_str or diagnosis_str == '-':
        return "진단 정보 없음"
    
    parts = diagnosis_str.split(' / ')
    formatted_parts = []
    
    for part in parts:
        if '(' in part and ')' in part:
            category_end = part.find('(')
            category = part[:category_end]
            detail = part[category_end+1:-1]
            formatted_parts.append(f"<strong>{category}:</strong> {detail}")
        else:
            formatted_parts.append(part)
    
    return ' | '.join(formatted_parts)

def format_top_products(df):
    """해당 그룹 내에서 빈도가 높은 상위 2개 제품명을 백분율로 표시"""
    if '제품명' not in df.columns: return ""
    
    product_counts = df['제품명'].value_counts()
    total_count = len(df)
    
    if total_count == 0 or product_counts.empty: return ""
    
    top_products = product_counts.head(2)
    formatted_parts = []
    
    for product, count in top_products.items():
        percentage = (count / total_count) * 100
        if pd.notna(product) and str(product).strip():
            formatted_parts.append(f"{product}({percentage:.0f}%)")
    
    if formatted_parts: return " | ".join(formatted_parts)
    return ""

def format_trend_with_highlight(trend_str):
    """추이 문자열에서 마지막 숫자를 굵게 강조"""
    if not trend_str or trend_str == '-': return "-"
    parts = trend_str.split(' → ')
    if len(parts) <= 1: return f"{trend_str}"
    
    normal_parts = parts[:-1]
    last_part = parts[-1]
    normal_text = ' → '.join(normal_parts)
    highlighted_text = f'<strong style="color: #111827; background: #fef08a; padding: 0 4px; border-radius: 2px;">{last_part}</strong>'
    
    if normal_parts: return f'{normal_text} → {highlighted_text}'
    else: return highlighted_text

# --- 0. 페이지 설정 ---
st.set_page_config(
    page_title="Quality Control Tower",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [Global CSS] - Modern Dashboard Style with New Color System
st.markdown(f"""
<style>
    .block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; }}
    div[data-testid="stMetric"] {{
        background-color: white; padding: 15px; border: 1px solid #e5e7eb;
        border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .card-container {{
        background-color: white; border: 1px solid #e5e7eb;
        border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: box-shadow 0.3s;
    }}
    .card-container:hover {{ box-shadow: 0 10px 15px rgba(0,0,0,0.1); }}
    
    .risk-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }}
    .risk-title {{ font-size: 1.1rem; font-weight: 700; color: #111827; }}
    .risk-score {{ font-size: 1.5rem; font-weight: 800; }}
    
    .risk-badges {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
    .badge {{ padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }}
    .badge-gray {{ background: #f3f4f6; color: #374151; }}
    
    /* New Color System Applied */
    .badge-red {{ background: #fee2e2; color: {COLOR_RED}; }}
    .badge-yellow {{ background: #fff7ed; color: {COLOR_YELLOW}; }}
    .badge-blue {{ background: #e0f2fe; color: {COLOR_BLUE}; }}
    
    .risk-content {{ font-size: 0.9rem; color: #4b5563; line-height: 1.5; background: #f9fafb; padding: 10px; border-radius: 8px; }}
    .risk-footer {{ display: flex; justify-content: space-between; align-items: center; margin-top: 15px; padding-top: 10px; border-top: 1px solid #f3f4f6; }}
    
    .action-btn {{
        background: white; border: 1px solid #d1d5db; color: #374151;
        padding: 6px 12px; border-radius: 6px; font-size: 0.85rem; font-weight: 500;
        text-decoration: none; display: inline-flex; align-items: center; gap: 4px;
        transition: all 0.2s;
    }}
    .action-btn:hover {{ background: #f3f4f6; border-color: #9ca3af; color: #111827; text-decoration: none;}}
    .download-btn {{
        background: white; border: 1px solid #d1d5db; color: #374151;
        padding: 6px 12px; border-radius: 6px; font-size: 0.85rem; font-weight: 500;
        text-decoration: none; display: inline-flex; align-items: center; gap: 4px;
        transition: all 0.2s;
        margin-right: 4px;
    }}
    .download-btn:hover {{ background: #f3f4f6; border-color: #9ca3af; color: #111827; text-decoration: none;}}
</style>
""", unsafe_allow_html=True)

# --- 1. 데이터 로드 및 전처리 ---

def get_last_updated_time():
    try:
        if not DATA_HUB_PATH or not os.path.exists(DATA_HUB_PATH): return "-"
        root = Path(DATA_HUB_PATH)
        mtime = max(f.stat().st_mtime for f in root.rglob('*') if f.is_file())
        return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
    except:
        return datetime.now().strftime('%Y-%m-%d %H:%M')

@st.cache_data(ttl=3600)
def load_and_scan_risks_unified(mode='인입'):
    """Core Logic을 활용하여 데이터 로드 및 리스크 스캔 (Zero-Filling 적용)"""
    end_date = datetime.today().date()
    start_date = end_date - relativedelta(years=3)
    
    df = load_and_filter_data(
        plant="", # All plants
        start_date=start_date,
        end_date=end_date,
        search_mode=mode,
        data_path=DATA_HUB_PATH
    )
    
    if df.empty:
        try:
            dataset = ds.dataset(DATA_HUB_PATH, partitioning="hive", format="parquet")
            df = dataset.to_table().to_pandas()
        except:
            return None, None, None, None, None

    if df.empty: return None, None, None, None, None

    df['접수일자'] = pd.to_datetime(df['접수일자'])
    
    # 모드 필터링 (Core 로직 재현)
    if mode == '실적':
        perform_reasons = ['고객불만족', '구매불만', '제조불만']
        cond_reason = df['불만원인'].isin(perform_reasons)
        target_biz = ['식품', 'B2B식품']
        cond_biz = df['사업부문'].isin(target_biz) if '사업부문' in df.columns else True
        df = df[cond_reason & cond_biz].copy()
    elif mode == '인입':
        target_biz = ['식품', 'B2B식품']
        cond_biz = df['사업부문'].isin(target_biz) if '사업부문' in df.columns else True
        cond_reason = df['불만원인'].notna()
        df = df[cond_reason & cond_biz].copy()

    if '건수' not in df.columns: df['건수'] = 1
    
    # 기준일 설정 (전체 데이터 중 가장 최신 날짜)
    max_date = df['접수일자'].max()
    max_date_str = max_date.strftime('%Y-%m-%d')
    target_month_str = max_date.strftime('%Y-%m')
    prev_month_str = (max_date.replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
    
    # 2. Risk Data Preparation (Zero-Filling)
    pivot_keys = ['플랜트', '대분류', '소분류', '등급기준']
    
    risk_start_date = max_date - relativedelta(months=24)
    df_risk_src = df[df['접수일자'] >= risk_start_date].copy()
    
    risk_pivot = prepare_risk_data(
        df=df_risk_src,
        pivot_keys=pivot_keys,
        target_date=max_date,
        lookback_months=24
    )
    
    # 3. Risk Scoring Loop
    risk_results = []
    last_date_map = df_risk_src.groupby(pivot_keys)['접수일자'].max()
    
    for idx in risk_pivot.index:
        plant, cat_main, cat_sub, grade = idx
        series = risk_pivot.loc[idx] 
        
        # Risk Score Calculation
        status, score, reason = calculate_advanced_risk_score(series, target_month_str, grade=grade)
        
        if score > 0:
            last_date_val = last_date_map.get(idx, pd.NaT)
            last_date_str = last_date_val.strftime('%Y-%m-%d') if pd.notnull(last_date_val) else "-"
            
            # [FILTER] 최신 데이터인지 확인 (마지막 감지일 == 전체 데이터 최신일)
            if last_date_str == max_date_str:
                trend_vals = series.tail(6).astype(int).tolist()
                trend_str = " → ".join(map(str, trend_vals))
                
                risk_results.append({
                    '플랜트': plant,
                    '유형': f"{cat_main} > {cat_sub}",
                    '대분류': cat_main,
                    '소분류': cat_sub,
                    '등급': grade,
                    '건수': int(series.iloc[-1]), # 당월 건수
                    '상태': status,
                    '점수': score,
                    '진단': reason,
                    'Trend_Str': trend_str,
                    'Last_Date': last_date_str
                })
            
    risk_df = pd.DataFrame(risk_results)
    if not risk_df.empty:
        risk_df = risk_df.sort_values('점수', ascending=False)
        
    forecast_engine = ForecastEngine(df, date_col='접수일자')
    
    return df, risk_df, max_date_str, prev_month_str, forecast_engine

# --- 2. Dashboard Logic ---

# Sidebar
st.sidebar.markdown("### 📊 분석 모드")
selected_mode = st.sidebar.radio(
    "조회 모드 선택",
    options=["인입 (Inflow)", "실적 (Performance)"],
    horizontal=False
)
mode = '인입' if selected_mode == "인입 (Inflow)" else '실적'

# Cache Control
if 'prev_mode' not in st.session_state:
    st.session_state.prev_mode = mode
elif st.session_state.prev_mode != mode:
    st.cache_data.clear()
    st.session_state.prev_mode = mode

with st.spinner("📡 데이터 분석 및 리스크 스캔 중..."):
    # max_date_str를 받아서 표시
    raw_df, risk_report, current_date_str, prev_month, forecast_engine = load_and_scan_risks_unified(mode=mode)
    last_updated = get_last_updated_time()

if raw_df is None:
    st.error("데이터 로드 실패.")
    st.stop()

# Header
c1, c2 = st.columns([3, 1])
c1.title("📡 Quality Control Tower")
mode_label = f"【{selected_mode}】"
c1.caption(f"기준년월일: {current_date_str} | 전사 통합 모니터링 {mode_label}")
c2.markdown(f"<div style='text-align:right; padding-top:20px; color:gray;'>Last Update: {last_updated}</div>", unsafe_allow_html=True)

# KPI Section
max_date = raw_df['접수일자'].max()
day_of_month = max_date.day
current_month_start = max_date.replace(day=1)
prev_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
end_day_prev = min(day_of_month, (prev_month_start + relativedelta(months=1) - timedelta(days=1)).day)
prev_month_end = prev_month_start.replace(day=end_day_prev)

def get_kpi_dynamic(df, grade=None):
    df_curr = df[(df['접수일자'] >= current_month_start) & (df['접수일자'] <= max_date)]
    df_prev = df[(df['접수일자'] >= prev_month_start) & (df['접수일자'] <= prev_month_end)]
    
    if grade:
        curr = df_curr[df_curr['등급기준'] == grade].shape[0]
        past = df_prev[df_prev['등급기준'] == grade].shape[0]
    else:
        curr = df_curr.shape[0]
        past = df_prev.shape[0]
        
    mom = ((curr - past)/past * 100) if past > 0 else 0
    return curr, mom

total_v, total_m = get_kpi_dynamic(raw_df)
danger_v, danger_m = get_kpi_dynamic(raw_df, "위험")
crit_v, crit_m = get_kpi_dynamic(raw_df, "중대")
gen_v, gen_m = get_kpi_dynamic(raw_df, "일반")

def render_metric(col, label, val, delta):
    color = COLOR_RED if delta > 0 else COLOR_BLUE if delta < 0 else COLOR_GRAY
    col.markdown(f"""
    <div style='background:white; padding:16px; border:1px solid #e5e7eb; border-radius:12px;'>
        <div style='color:#6b7280; font-size:0.9rem; font-weight:600;'>{label}</div>
        <div style='color:{color}; font-size:1.8rem; font-weight:800; margin:4px 0;'>{val:,}건</div>
        <div style='color:{color}; font-size:0.95rem;'>{delta:+.1f}% (전월 동기)</div>
    </div>
    """, unsafe_allow_html=True)

st.subheader(f"📊 전사 클레임 현황 ({max_date.strftime('%Y-%m-%d')} 기준)")
k1, k2, k3, k4 = st.columns(4)
render_metric(k1, f"전사({mode})", total_v, total_m)
render_metric(k2, "위험", danger_v, danger_m)
render_metric(k3, "중대", crit_v, crit_m)
render_metric(k4, "일반", gen_v, gen_m)

st.divider()

# Chart & Insight Layout
col_chart, col_insight = st.columns([3, 2])

with col_chart:
    st.markdown("#### 📈 전사 트렌드 (3개년 & 4개월 예측)")
    with st.container(border=True, height=450):
        # Trend Data Prep
        trend = raw_df.groupby('접수일자').size().reset_index(name='건수')
        trend['Year'] = trend['접수일자'].dt.year
        trend['Month'] = trend['접수일자'].dt.month
        
        tgt_year = max_date.year
        df_this = trend[trend['Year'] == tgt_year].groupby('Month')['건수'].sum()
        df_last = trend[trend['Year'] == tgt_year-1].groupby('Month')['건수'].sum()
        df_before = trend[trend['Year'] == tgt_year-2].groupby('Month')['건수'].sum()
        
        fig = go.Figure()
        
        # Historical Lines
        fig.add_trace(go.Scatter(x=df_before.index, y=df_before.values, name=f"{tgt_year-2}", 
                                 line=dict(color='gray', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=df_last.index, y=df_last.values, name=f"{tgt_year-1}", 
                                 line=dict(color='skyblue', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=df_this.index, y=df_this.values, name=f"{tgt_year}", 
                                 mode='lines+markers', line=dict(color=COLOR_RED, width=3)))
        
        # Forecast (ForecastEngine 활용)
        try:
            fcst_res = forecast_engine.forecast_4m()
            future_map = fcst_res['future_4m']
            
            f_months = []
            f_vals = []
            f_text = []
            
            for d_str in sorted(future_map.keys()):
                m_num = int(d_str.split('-')[1])
                val = future_map[d_str]
                f_months.append(m_num)
                f_vals.append(val)
                f_text.append(f"{m_num}월 예측: {val:,}건")
            
            # [FIX] 실적 라인과 연결하지 않음 (독립적 렌더링)
            if f_months:
                fig.add_trace(go.Scatter(x=f_months, y=f_vals, name='4개월 예측',
                                         mode='markers+lines', # 라인은 예측 점들끼리만 연결
                                         line=dict(color=COLOR_YELLOW, width=2, dash='dot'),
                                         marker=dict(symbol='diamond', size=8, color=COLOR_YELLOW),
                                         hovertemplate='%{text}<extra></extra>', text=f_text))
        except:
            pass
            
        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(tickvals=list(range(1,13)), showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f3f4f6'),
            legend=dict(orientation="h", y=1.1),
            plot_bgcolor='white'
        )
        st.plotly_chart(fig, width='stretch')

with col_insight:
    st.markdown("#### ⚠️ Critical LOT Check (동일 제조일 3건+)")
    
    # LOT Logic
    start_dt = max_date - timedelta(days=90)
    df_lot = raw_df[raw_df['접수일자'] >= start_dt].copy()
    
    # Clean Mfg Date
    def clean_mfg(x):
        try:
            return pd.to_datetime(x, unit='ms') if isinstance(x, (int, float)) and x > 1e10 else pd.to_datetime(x)
        except: return pd.NaT
        
    df_lot['mfg_dt'] = df_lot['제조일자'].apply(clean_mfg)
    df_lot = df_lot.dropna(subset=['mfg_dt'])
    df_lot['mfg_str'] = df_lot['mfg_dt'].dt.strftime('%Y-%m-%d')
    
    lot_groups = df_lot.groupby(['플랜트', '제품명', '제품코드', '소분류', 'mfg_str']).agg(
        last_receipt=('접수일자', 'max'),
        count=('접수일자', 'size')
    ).reset_index()
    
    critical_lots = lot_groups[lot_groups['count'] >= 3].sort_values('last_receipt', ascending=False)
    
    with st.container(border=True, height=450):
        if critical_lots.empty:
            st.success("✅ 최근 3개월 내 중복 이슈 없음")
        else:
            for idx, row in critical_lots.iterrows():
                is_today = row['last_receipt'].strftime('%Y-%m-%d') == max_date.strftime('%Y-%m-%d')
                bg_color = "#fff7ed" if is_today else "white"
                border_color = COLOR_YELLOW if is_today else "#e5e7eb"
                
                # Excel Download Logic
                dl_df = df_lot[
                    (df_lot['플랜트'] == row['플랜트']) & (df_lot['제품코드'] == row['제품코드']) &
                    (df_lot['mfg_str'] == row['mfg_str'])
                ]
                buf = BytesIO()
                with pd.ExcelWriter(buf) as writer: dl_df.to_excel(writer, index=False)
                b64 = base64.b64encode(buf.getvalue()).decode()
                href = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"
                
                # [FIX] Deep Link Logic (Category = Major)
                cat_main = ""
                if '대분류' in dl_df.columns and not dl_df['대분류'].empty:
                    cat_main = dl_df['대분류'].iloc[0]
                
                target_cat = cat_main if cat_main else "All"
                link_url = f"/플랜트_분석?plant={row['플랜트']}&category={target_cat}&subcategory={row['소분류']}"
                
                st.markdown(f"""
                <div style='background:{bg_color}; border:1px solid {border_color}; border-radius:8px; padding:12px; margin-bottom:8px;'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                        <div style='font-weight:700; font-size:0.9rem;'>🏭 {row['플랜트']} | {row['제품명']}</div>
                        <div style='font-weight:800; color:{COLOR_RED};'>{row['count']}건</div>
                    </div>
                    <div style='font-size:0.8rem; color:#4b5563; margin-bottom:8px;'>
                        제조: {row['mfg_str']} | 최근접수: {row['last_receipt'].strftime('%Y-%m-%d')}
                    </div>
                    <div style='display:flex; gap:8px;'>
                        <a href="{href}" download="LOT_{row['mfg_str']}.xlsx" 
                           style='flex:1; text-align:center; background:white; border:1px solid #d1d5db; border-radius:4px; text-decoration:none; font-size:0.8rem; color:#374151; padding:4px;'>
                           📥 엑셀
                        </a>
                        <a href="{link_url}" target="_self"
                           style='flex:1; text-align:center; background:{COLOR_BLUE}; border:1px solid {COLOR_BLUE}; border-radius:4px; text-decoration:none; font-size:0.8rem; color:white; padding:4px;'>
                           🔬 분석
                        </a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.subheader("🚨 Risk Radar (Action Dashboard)")

if not risk_report.empty:
    cnt_r = risk_report[risk_report['상태']=='🔴'].shape[0]
    cnt_y = risk_report[risk_report['상태']=='🟡'].shape[0]
    
    c_red, c_yel = st.columns(2)
    
    # Calculate global download start date for risk radar (6 months)
    risk_download_start_date = max_date - relativedelta(months=6)

    def render_risk_cards(container, df, color_theme):
        if df.empty:
            container.success("해당 등급의 리스크가 없습니다.")
            return
            
        for _, row in df.iterrows():
            # Styling vars
            score_color = COLOR_RED if color_theme == 'red' else COLOR_YELLOW
            border_color = "#fecaca" if color_theme == 'red' else "#fde68a"
            badge_class = "badge-red" if color_theme == 'red' else "badge-yellow"
            
            # Top Products info (using Product Name)
            top_prod_info = ""
            
            # Filter raw_df for this risk group (for both top products and download)
            group_df = raw_df[
                (raw_df['플랜트'] == row['플랜트']) &
                (raw_df['대분류'] == row['대분류']) &
                (raw_df['소분류'] == row['소분류'])
            ]
            top_prod_info = format_top_products(group_df)
            
            # [NEW] Excel Download for Risk Radar (Recent 6 months)
            download_df = group_df[group_df['접수일자'] >= risk_download_start_date]
            buf = BytesIO()
            with pd.ExcelWriter(buf) as writer: download_df.to_excel(writer, index=False)
            b64 = base64.b64encode(buf.getvalue()).decode()
            excel_href = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"
            
            # Link Logic
            link_url = f"/플랜트_분석?plant={row['플랜트']}&grade={row['등급']}&category={row['대분류']}&subcategory={row['소분류']}"
            pred_url = f"/예측_시뮬레이션?plant={row['플랜트']}&category={row['대분류']}"
            
            container.markdown(f"""
            <div class='card-container' style='border-left: 4px solid {score_color}; margin-bottom: 16px;'>
                <div class='risk-header'>
                    <div>
                        <div class='risk-title'>🏭 {row['플랜트']} - {row['소분류']}</div>
                        <div class='risk-badges'>
                            <span class='badge {badge_class}'>{row['등급']}</span>
                            <span class='badge badge-gray'>{row['대분류']}</span>
                            <span class='badge badge-blue'>당월 {row['건수']}건</span>
                        </div>
                    </div>
                    <div class='risk-score' style='color:{score_color}'>{row['점수']}점</div>
                </div>
                <div class='risk-content'>
                    <strong>💡 진단:</strong> {format_diagnosis(row['진단'])}<br>
                    <strong>📦 주요 제품:</strong> {top_prod_info}<br>
                    <strong>📈 추이:</strong> {format_trend_with_highlight(row['Trend_Str'])}
                </div>
                <div class='risk-footer'>
                    <span style='font-size:0.8rem; color:#6b7280;'>최근감지: {row['Last_Date']}</span>
                    <div>
                        <a href="{excel_href}" download="Risk_{row['플랜트']}_{row['소분류']}.xlsx" class='download-btn'>📥 엑셀</a>
                        <a href="{link_url}" target="_self" class='action-btn' style='margin-right:4px;'>🔬 정밀</a>
                        <a href="{pred_url}" target="_self" class='action-btn'>🔮 예측</a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with c_red:
        st.markdown(f"#### 🔴 Danger ({cnt_r}건)")
        with st.container(height=600, border=True):
            render_risk_cards(st, risk_report[risk_report['상태']=='🔴'], 'red')
            
    with c_yel:
        st.markdown(f"#### 🟡 Caution ({cnt_y}건)")
        with st.container(height=600, border=True):
            render_risk_cards(st, risk_report[risk_report['상태']=='🟡'], 'yellow')

else:
    st.success("🎉 현재 감지된 리스크가 없습니다. 안정적입니다.")
