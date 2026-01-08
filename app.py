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

# [Core Engine] Phase 2.8 엔진 탑재
from core.storage import DATA_HUB_PATH
from core.analytics import calculate_advanced_risk_score

# --- 0. Helper Functions ---
def format_diagnosis(diagnosis_str):
    """진단 결과를 category와 detail로 분리해서 표시"""
    if not diagnosis_str or diagnosis_str == '-':
        return "진단 정보 없음"
    
    # Parse the diagnosis string (e.g., "⚡돌발감지(희소유형 발생 감지) / 📊추세이탈(패턴 이탈 감지)")
    parts = diagnosis_str.split(' / ')
    formatted_parts = []
    
    for part in parts:
        if '(' in part and ')' in part:
            # Split category and detail
            category_end = part.find('(')
            category = part[:category_end]
            detail = part[category_end+1:-1]  # Remove parentheses
            formatted_parts.append(f"<strong>{category}:</strong> {detail}")
        else:
            formatted_parts.append(part)
    
    return ' | '.join(formatted_parts)

def format_product_categories(df):
    """제품범주2의 상위 카테고리들을 백분율로 표시"""
    if '제품범주2' not in df.columns:
        return ""
    
    # 제품범주2별 건수 계산
    category_counts = df['제품범주2'].value_counts()
    total_count = len(df)
    
    if total_count == 0 or category_counts.empty:
        return ""
    
    # 상위 2개 카테고리 선택
    top_categories = category_counts.head(2)
    
    # 백분율 계산 및 포맷팅
    formatted_parts = []
    for category, count in top_categories.items():
        percentage = (count / total_count) * 100
        if pd.notna(category) and str(category).strip():
            formatted_parts.append(f"{category}({percentage:.0f}%)")
    
    if formatted_parts:
        return " | ".join(formatted_parts)
    return ""

def format_trend_with_highlight(trend_str):
    """추이 문자열에서 마지막 숫자를 굵게 강조(검은색)

    Returns the sequence with the final value wrapped in <strong> tags (black).
    """
    if not trend_str or trend_str == '-':
        return "추이 정보 없음"

    # "1 → 2 → 3 → 4 → 5 → 6" 형식 파싱
    parts = trend_str.split(' → ')
    if len(parts) <= 1:
        return f"{trend_str}"

    # 마지막 부분을 제외한 나머지
    normal_parts = parts[:-1]
    last_part = parts[-1]

    # 마지막 숫자를 굵게(검은색)로 표시
    normal_text = ' → '.join(normal_parts)
    highlighted_text = f'<strong style="color: #111827;">{last_part}</strong>'

    if normal_parts:
        return f'{normal_text} → {highlighted_text}'
    else:
        return highlighted_text

# --- 0. 페이지 설정 ---
st.set_page_config(
    page_title="Quality Control Tower",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [Global CSS]
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
    div[data-testid="stMetric"] {
        background-color: white; padding: 10px; border: 1px solid #e5e7eb;
        border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    div[data-testid="stButton"] button { width: 100%; }
    .lot-card-container {
        background-color: white; border: 1px solid #e5e7eb;
        border-radius: 8px; padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .lot-row { display: flex; gap: 12px; align-items: stretch; margin-bottom: 12px; }
    .lot-card-left { flex: 5; }
    .lot-download-box { flex: 1; display: flex; align-items: stretch; }
    .lot-download-btn {
        width: 100%; height: 100%;
        background-color: white; color: #111827;
        border-radius: 8px; border: 1px solid #e5e7eb;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; font-weight: 600; text-decoration: none;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .lot-download-btn:hover { background-color: #f9fafb; color: #111827; text-decoration: none; }
    .lot-title { font-weight: 600; color: #111827; font-size: 0.95rem; margin-bottom: 8px; }
    .lot-info { color: #111827; font-size: 0.95rem; line-height: 1.6; }
    .lot-count-badge {
        background-color: #dc2626; color: white; padding: 4px 12px;
        border-radius: 12px; font-size: 0.8rem; font-weight: 600;
        display: inline-block;
    }
    .lot-category-badge {
        background-color: #f3f4f6; color: #111827; padding: 4px 12px;
        border-radius: 12px; font-size: 0.8rem; font-weight: 600;
        display: inline-block;
    }
    .lot-product-badge {
        background-color: #e0f2fe; color: #0277bd; padding: 4px 12px;
        border-radius: 12px; font-size: 0.75rem; font-weight: 500;
        display: inline-block; border: 1px solid #b3e5fc;
    }
    .lot-grade-badge {
        padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;
        display: inline-flex; align-items: center; margin-left: 8px; vertical-align: middle;
        box-sizing: border-box;
    }
    .grade-normal { background:#fef3c7; color:#92400e; }
    .grade-unclassified { background:#f3f4f6; color:#6b7280; }
    .grade-danger { background:#fee2e2; color:#991b1b; }
</style>
""", unsafe_allow_html=True)

# --- 0-B. 고급 월말 예측 함수 ---

def predict_month_end_advanced(df_this: pd.Series, df_last: pd.Series, df_before_last: pd.Series, 
                                 current_month: int, days_passed: int, days_in_month: int) -> dict:
    
    result = {
        'predicted_val': 0.0,
        'method': '미결정',
        'yoy_baseline': 0.0,
        'trend_slope_yoy': 0.0,
        'trend_slope_recent': 0.0,
        'confidence': '낮음',
    }
    
    if days_passed <= 0:
        return result
    
    try:
        # Step 1: 동월 YoY 데이터 수집
        yoy_current = df_this.get(current_month, 0)
        yoy_last = df_last.get(current_month, 0) if len(df_last) > 0 else 0
        yoy_before = df_before_last.get(current_month, 0) if len(df_before_last) > 0 else 0
        
        # Step 2: 추세선 계산
        trend_slope_yoy = 0.0
        if len(df_last) >= 3:
            x_yoy = np.arange(len(df_last), dtype=float)
            y_yoy = np.array(df_last.values, dtype=float)
            z_yoy = np.polyfit(x_yoy, y_yoy, 1)
            trend_slope_yoy = float(z_yoy[0])
        
        trend_slope_recent = 0.0
        recent_months_data = []
        for m in range(max(1, current_month - 2), current_month):
            if m in df_this.index:
                recent_months_data.append(df_this[m])
        
        if len(recent_months_data) >= 2:
            x_recent = np.arange(len(recent_months_data), dtype=float)
            y_recent = np.array(recent_months_data, dtype=float)
            z_recent = np.polyfit(x_recent, y_recent, 1)
            trend_slope_recent = float(z_recent[0])
        
        # Step 3: 예측값 계산
        current_val = yoy_current
        daily_avg_baseline = current_val / days_passed if days_passed > 0 else 0
        
        pred_baseline = daily_avg_baseline * days_in_month
        
        pred_yoy = yoy_last
        if abs(trend_slope_yoy) > 0.001:
            pred_yoy += trend_slope_yoy * (days_passed / days_in_month)
        
        pred_2y = yoy_before
        if abs(trend_slope_recent) > 0.001:
            pred_2y += trend_slope_recent * (days_passed / days_in_month)
        
        valid_yoy_years = sum([1 for v in [yoy_current, yoy_last, yoy_before] if v > 0])
        
        if valid_yoy_years >= 2:
            weights = [0.4, 0.4, 0.2]
            predictions = [pred_baseline, pred_yoy, pred_2y]
        else:
            weights = [0.7, 0.2, 0.1]
            predictions = [pred_baseline, pred_yoy, pred_2y]
        
        predicted_val = sum(w * p for w, p in zip(weights, predictions))
        predicted_val = max(0, predicted_val)
        
        # Step 4: 신뢰도 판단
        has_yoy = yoy_last > 0
        has_2y = yoy_before > 0
        has_trend = abs(trend_slope_yoy) > 0.001 or abs(trend_slope_recent) > 0.001
        
        if (has_yoy and has_2y) and has_trend:
            confidence = '높음'
        elif has_yoy or (has_2y and has_trend):
            confidence = '중간'
        else:
            confidence = '낮음'
        
        result = {
            'predicted_val': float(predicted_val),
            'method': f'YoY 가중 예측 (신뢰도: {confidence})',
            'yoy_baseline': float(yoy_last),
            'trend_slope_yoy': float(trend_slope_yoy),
            'trend_slope_recent': float(trend_slope_recent),
            'confidence': confidence,
        }
        
        return result
    
    except Exception as e:
        print(f"[WARNING] 예측 계산 실패: {e}")
        return result


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
def load_and_scan_risks():
    try:
        if not DATA_HUB_PATH: return None, None, None, None
        dataset = ds.dataset(DATA_HUB_PATH, partitioning="hive", format="parquet")
        df = dataset.to_table().to_pandas()
        if df.empty: return None, None, None, None
    except:
        return None, None, None, None

    df['접수일자'] = pd.to_datetime(df['접수일자'])
    df['접수월'] = df['접수일자'].dt.strftime('%Y-%m')
    
    max_date = df['접수일자'].max()
    target_month = max_date.strftime('%Y-%m')
    prev_month_date = max_date.replace(day=1) - timedelta(days=1)
    prev_month = prev_month_date.strftime('%Y-%m')

    # 1. Pivot Table (건수 집계)
    grouped = df.groupby(['플랜트', '대분류', '소분류', '등급기준', '접수월']).size().reset_index(name='건수')
    pivot = grouped.pivot_table(index=['플랜트', '대분류', '소분류', '등급기준'], columns='접수월', values='건수', fill_value=0)
    
    # 2. Last Date Map
    last_date_series = df.groupby(['플랜트', '대분류', '소분류', '등급기준'])['접수일자'].max()
    
    risk_results = []
    if target_month not in pivot.columns: return df, pd.DataFrame(), target_month, prev_month

    targets = pivot.index
    date_cols = sorted([c for c in pivot.columns if isinstance(c, str) and c.startswith('20')])

    for idx in targets:
        plant, cat_main, cat_sub, grade = idx
        series = pivot.loc[idx, date_cols]
        try: current_val = int(series[target_month])
        except: current_val = 0
            
        status, score, reason = calculate_advanced_risk_score(series, target_month, grade=grade)
        
        if score > 0:
            trend_list = series.tolist()[-6:]
            trend_str = " → ".join([str(int(x)) for x in trend_list])
            
            last_date_val = last_date_series.get(idx, pd.NaT)
            last_date_str = last_date_val.strftime('%Y-%m-%d') if pd.notnull(last_date_val) else "-"
            
            risk_results.append({
                '플랜트': plant,
                '유형': f"{cat_main} > {cat_sub}",
                '대분류': cat_main,
                '등급': grade,
                '건수': current_val,
                '상태': status,
                '점수': score,
                '진단': reason,
                'Trend_Str': trend_str,
                'Last_Date': last_date_str
            })
            
    risk_df = pd.DataFrame(risk_results)
    if not risk_df.empty: risk_df = risk_df.sort_values('점수', ascending=False)
        
    return df, risk_df, target_month, prev_month

# --- 2. Dashboard Logic ---

with st.spinner("📡 데이터 분석 중..."):
    raw_df, risk_report, target_month, prev_month = load_and_scan_risks()
    last_updated = get_last_updated_time()

if raw_df is None:
    st.error("데이터가 없습니다.")
    st.stop()

# [Header]
c1, c2 = st.columns([3, 1])
c1.title("📡 Quality Control Tower")
c1.caption(f"기준년월: {target_month} | 전사 통합 모니터링")
c2.markdown(f"<div style='text-align:right; padding-top:20px; color:gray;'>Last Update: {last_updated}</div>", unsafe_allow_html=True)

# [KPI]
max_date = raw_df['접수일자'].max()
day_of_month = max_date.day

current_month_start = max_date.replace(day=1)
prev_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
end_day_prev_month = min(day_of_month, pd.Timestamp(prev_month_start).days_in_month)
prev_month_end = prev_month_start.replace(day=end_day_prev_month)

def get_kpi_dynamic(df, grade=None):
    df_current = df[(df['접수일자'] >= current_month_start) & (df['접수일자'] <= max_date)]
    df_prev = df[(df['접수일자'] >= prev_month_start) & (df['접수일자'] <= prev_month_end)]

    if grade:
        curr = df_current[df_current['등급기준']==grade].shape[0]
        past = df_prev[df_prev['등급기준']==grade].shape[0]
    else:
        curr = df_current.shape[0]
        past = df_prev.shape[0]
    mom = ((curr - past)/past * 100) if past > 0 else 0
    return curr, mom

total_v, total_m = get_kpi_dynamic(raw_df)
danger_v, danger_m = get_kpi_dynamic(raw_df, "위험")
crit_v, crit_m = get_kpi_dynamic(raw_df, "중대")
gen_v, gen_m = get_kpi_dynamic(raw_df, "일반")

st.subheader(f"📊 전사 클레임 인입 현황 ({max_date.strftime('%Y/%m/%d')} 기준)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("전사 Total", f"{total_v:,}건", f"{total_m:+.1f}% (전월 동기 비)", delta_color="off" if total_m == 0 else "inverse")
k2.metric("위험 등급", f"{danger_v:,}건", f"{danger_m:+.1f}% (전월 동기 비)", delta_color="off" if danger_m == 0 else "inverse")
k3.metric("중대 등급", f"{crit_v:,}건", f"{crit_m:+.1f}% (전월 동기 비)", delta_color="off" if crit_m == 0 else "inverse")
k4.metric("일반 등급", f"{gen_v:,}건", f"{gen_m:+.1f}% (전월 동기 비)", delta_color="off" if gen_m == 0 else "inverse")

st.divider()

# [Chart & Insight] - Equal two-column layout for trend and LOT
col_chart, col_insight = st.columns([3, 2])
with col_chart:
    st.markdown("#### 📈 전사 트렌드 (3개년)")
    with st.container(border=True, height=450):
        trend = raw_df.groupby('접수일자').size().reset_index(name='건수')
        trend['Year'] = trend['접수일자'].dt.year
        trend['Month'] = trend['접수일자'].dt.month
        tgt_year = datetime.strptime(target_month, "%Y-%m").year
        df_this = trend[trend['Year'] == tgt_year].groupby('Month')['건수'].sum()
        df_last = trend[trend['Year'] == tgt_year-1].groupby('Month')['건수'].sum()
        df_before_last = trend[trend['Year'] == tgt_year-2].groupby('Month')['건수'].sum()
        
        fig = go.Figure()
        hovertemp = "<b>%{meta}년 %{x}</b><br>건수 : %{y:,}건<extra></extra>"

        fig.add_trace(go.Scatter(x=df_before_last.index, y=df_before_last.values, name=f"{tgt_year-2}",
                                 meta=tgt_year-2, hovertemplate=hovertemp,
                                 line=dict(color='gray', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=df_last.index, y=df_last.values, name=f"{tgt_year-1}",
                                 meta=tgt_year-1, hovertemplate=hovertemp,
                                 line=dict(color='skyblue', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=df_this.index, y=df_this.values, name=f"{tgt_year}",
                                 meta=tgt_year, hovertemplate=hovertemp,
                                 mode='lines+markers', fill='tozeroy', line=dict(color='#ef4444', width=3)))
        
        max_date_data = raw_df['접수일자'].max()
        current_month = max_date_data.month
        days_passed = max_date_data.day
        days_in_month = max_date_data.days_in_month
        
        if days_passed < days_in_month:
            current_val = df_this.get(current_month, 0)
            if days_passed > 0:
                pred_result = predict_month_end_advanced(
                    df_this=df_this, df_last=df_last, df_before_last=df_before_last,
                    current_month=current_month, days_passed=days_passed, days_in_month=days_in_month
                )
                predicted_val = pred_result['predicted_val']
                
                fig.add_trace(go.Scatter(
                    x=[current_month], y=[predicted_val], name='월말 예측',
                    mode='markers+text',
                    marker=dict(color='yellow', size=12, symbol='circle', line=dict(width=2, color='darkred')),
                    text=[f"{predicted_val:.0f}"], textposition="top center", textfont=dict(size=14, color='#ef4444'),
                    hovertext=f"<b>월말 예측값: {predicted_val:.0f}건</b><br>현재값: {current_val:.0f}건<br>방식: {pred_result['method']}",
                    hoverinfo='text'
                ))

        fig.update_layout(
            height=350, margin=dict(l=10, r=10, t=10, b=10), 
            xaxis=dict(tickvals=list(range(1, 13)), ticktext=[f"{i}월" for i in range(1, 13)], range=[0.5, 12.5], showgrid=False), 
            yaxis=dict(title_text="(건수)", showgrid=True, gridcolor='#f3f4f6'), 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
            plot_bgcolor='white'
        )
        st.plotly_chart(fig, width='stretch')

# [NEW: Critical LOT Check]
with col_insight:
    st.markdown("#### ⚠️ 주요 점검필요 LOT(동일 제조일 3건 이상 발생)")
    
    # 1. 최근 1개월(롤링 30일) 데이터 필터링
    # rolling window: 최근 30일(포함) 기준으로 필터링
    max_date = raw_df['접수일자'].max()
    start_dt = max_date - timedelta(days=30)
    
    # Data Cleaning & Conversion
    df_lot = raw_df[raw_df['접수일자'] >= start_dt].copy()
    
    # 제조일자 처리 (숫자/문자 혼용 대응)
    def clean_mfg_date(val):
        try:
            if pd.isna(val): return pd.NaT
            # 숫자인 경우 (timestamp ms)
            if isinstance(val, (int, float)):
                if val > 1000000000000: # ms timestamp assumed
                    return pd.to_datetime(val, unit='ms')
            # 문자인 경우
            return pd.to_datetime(val, errors='coerce')
        except:
            return pd.NaT

    df_lot['mfg_dt'] = df_lot['제조일자'].apply(clean_mfg_date)
    df_lot = df_lot.dropna(subset=['mfg_dt'])
    df_lot['mfg_str'] = df_lot['mfg_dt'].dt.strftime('%Y-%m-%d')
    
    # 2. Grouping (플랜트 | 제품명 | 제품코드 | 소분류 | 제조일자)
    # count >= 3 필터링, 접수일자 기준 내림차순 정렬
    lot_groups = df_lot.groupby(['플랜트', '제품명', '제품코드', '소분류', 'mfg_str']).agg(
        last_receipt=('접수일자', 'max'),  # 가장 최근 접수일자
        count=('접수일자', 'size')          # 건수
    ).reset_index()
    lot_groups['last_receipt_str'] = pd.to_datetime(lot_groups['last_receipt']).dt.strftime('%Y-%m-%d')
    critical_lots = lot_groups[lot_groups['count'] >= 3].sort_values('last_receipt', ascending=False)
    
    # 3. Rendering
    with st.container(border=True, height=450):
        if critical_lots.empty:
            st.success("✅ 최근 3개월 내 동일 제조일자 3건 이상 중복된 이슈가 없습니다.")
        else:
            st.markdown(f"<div style='color:#111827; font-weight:500;'>· 최근 1개월 ({start_dt.strftime('%Y-%m-%d')}~{max_date.strftime('%Y-%m-%d')})</div>", unsafe_allow_html=True)
            
            for idx, row in critical_lots.iterrows():
                is_recent = row['last_receipt_str'] == max_date.strftime('%Y-%m-%d')
                # Excel 다운로드 준비
                download_data = df_lot[
                    (df_lot['플랜트'] == row['플랜트']) &
                    (df_lot['제품명'] == row['제품명']) &
                    (df_lot['제품코드'] == row['제품코드']) &
                    (df_lot['소분류'] == row['소분류']) &
                    (df_lot['mfg_str'] == row['mfg_str'])
                ]
                code_str = f"({int(row['제품코드'])})" if pd.notna(row['제품코드']) else ""
                # 등급기준 추출: 그룹 내 가장 빈도가 높은 값 사용
                grade_val = "미분류"
                grade_css = "grade-unclassified"
                if '등급기준' in download_data.columns and not download_data['등급기준'].dropna().empty:
                    try:
                        g = download_data['등급기준'].mode().iloc[0]
                        grade_val = str(g) if pd.notna(g) and str(g).strip() != '' else '미분류'
                        if grade_val == '일반':
                            grade_css = 'grade-normal'
                        elif grade_val.strip() == '':
                            grade_val = '미분류'
                            grade_css = 'grade-unclassified'
                        else:
                            grade_css = 'grade-danger'
                    except Exception:
                        grade_val = '미분류'
                        grade_css = 'grade-unclassified'
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    download_data.to_excel(writer, index=False, sheet_name='LOT Details')
                excel_data = buffer.getvalue()
                download_b64 = base64.b64encode(excel_data).decode()
                download_href = (
                    "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," + download_b64
                )

                # 단일 flex-row HTML로 좌측 카드와 우측 버튼 높이 일치
                st.markdown(f"""
                <div class='lot-row'>
                    <div class='lot-card-container lot-card-left' style='{"background-color: #faf3f0;" if is_recent else ""}'>
                        <div class='lot-title'>🏭 {row['플랜트']} - {code_str}{row['제품명']} </div>
                        <div class='lot-info'>
                            📦 소분류: {row['소분류']}
                            {f"<span class='lot-grade-badge {grade_css}'>{grade_val}</span>" if grade_val or grade_css else ''}
                            &nbsp;&nbsp;
                            <span class='lot-count-badge'>{row['count']}건</span>
                        </div>
                        <div class='lot-info' style='margin-top: 8px;'>
                            🏷️ 제조일자: {row['mfg_str']} &nbsp;|&nbsp; 
                            📅 최근 접수: {row['last_receipt_str']}
                        </div>
                    </div>
                    <div class='lot-download-box'>
                        <a href="{download_href}" download="LOT_{row['플랜트']}_{row['mfg_str']}.xlsx" class='lot-download-btn'>📥엑셀</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# [D] Risk Radar (Interactive & Expanded) - Modern Card Style
st.subheader("🚨 Risk Radar (당월 이슈 신속경보)")

if not risk_report.empty:
    cnt_r = risk_report[risk_report['상태']=='🔴'].shape[0]
    cnt_y = risk_report[risk_report['상태']=='🟡'].shape[0]
    
    # 추이 데이터와 동일한 최근 6개월 기간 계산 (엑셀 다운로드 필터링용)
    max_date = raw_df['접수일자'].max()
    trend_start_date = (max_date.replace(day=1) - relativedelta(months=5)).replace(day=1)
    
    c_red, c_yellow = st.columns(2)
    
    # === 🔴 Danger Column ===
    with c_red:
        st.markdown(f":red[**🔴 위험 경보 (Danger) - {cnt_r}건**]")
        with st.container(height=800, border=True): 
            red_df = risk_report[risk_report['상태']=='🔴']
            if red_df.empty:
                st.success("위험 등급 이슈가 없습니다.")
            else:
                for idx, row in red_df.iterrows():
                    is_recent = row['Last_Date'] == max_date.strftime('%Y-%m-%d')
                    # 1. Data Preparation
                    try:
                        cat_sub = row['유형'].split(' > ')[1] if isinstance(row['유형'], str) and '>' in row['유형'] else ''
                    except Exception:
                        cat_sub = ''
                    
                    download_df = raw_df[
                        (raw_df['플랜트'] == row['플랜트']) &
                        (raw_df['대분류'] == row['대분류']) &
                        (raw_df['소분류'] == cat_sub) &
                        (raw_df['접수일자'] >= trend_start_date)  # 추이 데이터와 동일한 최근 6개월 필터링
                    ]

                    # Build Excel bytes
                    buf = BytesIO()
                    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                        download_df.to_excel(writer, index=False, sheet_name='Risk Details')
                    excel_bytes = buf.getvalue()
                    excel_b64 = base64.b64encode(excel_bytes).decode()
                    excel_href = "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," + excel_b64

                    # 제품범주2 정보 계산
                    product_info = format_product_categories(download_df)

                    # 2. Grade Badge Logic
                    grade_display = row.get('등급', '') if pd.notna(row.get('등급', '')) else '미분류'
                    if grade_display == '일반':
                        grade_css = 'grade-normal'
                    elif grade_display == '미분류':
                        grade_css = 'grade-unclassified'
                    else:
                        grade_css = 'grade-danger'

                    # 3. Score Color (Danger = red)
                    score_color = "#dc2626"

                    # 4. Render as Modern Card with matched heights
                    st.markdown(f"""
                    <div style='display: flex; gap: 10px; margin-bottom: 12px;'>
                      <div style='flex: 0.8;'>
                        <div class='lot-card-container' style='display: flex; flex-direction: column; gap: 8px; margin-bottom: 0; {"background-color: #faf3f0;" if is_recent else ""}'>
                            <div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;'>
                                <div style='display: flex; align-items: center; gap: 12px;'>
                                    <div class='lot-title' style='margin-bottom: 0;'>🏭 {row['플랜트']}</div>
                                    <div style='font-size: 0.85rem; color: #374151; font-weight: 500; background: #f9fafb; padding: 4px 8px; border-radius: 4px; border-left: 3px solid #dc2626;'>{format_diagnosis(row.get('진단', '-'))}</div>
                                </div>
                                <div style='font-weight: 800; font-size: 1.3rem; color: {score_color};'>{int(row['점수'])}점</div>
                            </div>
                            <div style='display: flex; gap: 8px; align-items: center; flex-wrap: wrap;'>
                                <span class='lot-grade-badge {grade_css}'>{grade_display}</span>
                                <span class='lot-category-badge'>{row['대분류']} > {cat_sub}</span>                                
                                <span class='lot-count-badge'>{int(row['건수'])}건</span>
                                {f"<span class='lot-product-badge'>주요 제품군: {product_info}</span>" if product_info else ""}
                            </div>
                            <div class='lot-info' style='margin-top: 6px; padding-top: 6px; border-top: 1px solid #f3f4f6;'>
                                <div style='display:flex; justify-content:space-between; align-items:center; font-size:0.95rem; color:#111827;'>
                                    <div>📈 추이: {format_trend_with_highlight(row['Trend_Str'])}</div>
                                    <div style='font-size:0.9rem; color:#111827;'>🔍감지일자 : {row['Last_Date']}</div>
                                </div>
                            </div>
                        </div>
                      </div>
                      <div style='flex: 0.2; display: flex; flex-direction: column; gap: 8px;'>
                        <a href="{excel_href}" download="Risk_{row['플랜트']}_{cat_sub}.xlsx" class='lot-download-btn' style='flex: 1;'>📥엑셀</a>
                        <a href="pages/3_플랜트_분석.py?plant={row['플랜트']}&grade={row['등급']}&category={row['대분류']}" class='lot-download-btn' style='flex: 1;'>🔬분석</a>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

    # === 🟡 Caution Column ===
    with c_yellow:
        st.markdown(f":orange[**🟡 주의 경보 (Caution) - {cnt_y}건**]")
        with st.container(height=800, border=True): 
            yellow_df = risk_report[risk_report['상태']=='🟡']
            if yellow_df.empty:
                st.success("주의 등급 이슈가 없습니다.")
            else:
                for idx, row in yellow_df.iterrows():
                    is_recent = row['Last_Date'] == max_date.strftime('%Y-%m-%d')
                    # 1. Data Preparation
                    try:
                        cat_sub = row['유형'].split(' > ')[1] if isinstance(row['유형'], str) and '>' in row['유형'] else ''
                    except Exception:
                        cat_sub = ''
                    
                    download_df = raw_df[
                        (raw_df['플랜트'] == row['플랜트']) &
                        (raw_df['대분류'] == row['대분류']) &
                        (raw_df['소분류'] == cat_sub) &
                        (raw_df['접수일자'] >= trend_start_date)  # 추이 데이터와 동일한 최근 6개월 필터링
                    ]

                    buf = BytesIO()
                    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                        download_df.to_excel(writer, index=False, sheet_name='Risk Details')
                    excel_bytes = buf.getvalue()
                    excel_b64 = base64.b64encode(excel_bytes).decode()
                    excel_href = "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," + excel_b64

                    # 제품범주2 정보 계산
                    product_info = format_product_categories(download_df)

                    # 2. Grade Badge Logic
                    grade_display = row.get('등급', '') if pd.notna(row.get('등급', '')) else '미분류'
                    if grade_display == '일반':
                        grade_css = 'grade-normal'
                    elif grade_display == '미분류':
                        grade_css = 'grade-unclassified'
                    else:
                        grade_css = 'grade-danger'

                    # 3. Score Color (Caution = orange/amber)
                    score_color = "#f59e0b"

                    # 4. Render as Modern Card with matched heights
                    st.markdown(f"""
                    <div style='display: flex; gap: 10px; margin-bottom: 12px;'>
                      <div style='flex: 0.8;'>
                        <div class='lot-card-container' style='display: flex; flex-direction: column; gap: 8px; margin-bottom: 0; {"background-color: #faf3f0;" if is_recent else ""}'>
                            <div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;'>
                                <div style='display: flex; align-items: center; gap: 12px;'>
                                    <div class='lot-title' style='margin-bottom: 0;'>🏭 {row['플랜트']}</div>
                                    <div style='font-size: 0.85rem; color: #374151; font-weight: 500; background: #f9fafb; padding: 4px 8px; border-radius: 4px; border-left: 3px solid #f59e0b;'>{format_diagnosis(row.get('진단', '-'))}</div>
                                </div>
                                <div style='font-weight: 800; font-size: 1.3rem; color: {score_color};'>{int(row['점수'])}점</div>
                            </div>
                            <div style='display: flex; gap: 8px; align-items: center; flex-wrap: wrap;'>
                                <span class='lot-grade-badge {grade_css}'>{grade_display}</span>
                                <span class='lot-category-badge'>{row['대분류']} > {cat_sub}</span>
                                <span class='lot-count-badge'>{int(row['건수'])}건</span>
                                {f"<span class='lot-product-badge'>주요 제품군: {product_info}</span>" if product_info else ""}
                            </div>
                            <div class='lot-info' style='margin-top: 6px; padding-top: 6px; border-top: 1px solid #f3f4f6;'>
                                <div style='display:flex; justify-content:space-between; align-items:center; font-size:0.95rem; color:#111827;'>
                                    <div>📈 추이: {format_trend_with_highlight(row['Trend_Str'])}</div>
                                    <div style='font-size:0.9rem; color:#111827;'>🔍감지일자 : {row['Last_Date']}</div>
                                </div>
                            </div>
                        </div>
                      </div>
                      <div style='flex: 0.2; display: flex; flex-direction: column; gap: 8px;'>
                        <a href="{excel_href}" download="Risk_{row['플랜트']}_{cat_sub}.xlsx" class='lot-download-btn' style='flex: 1;'>📥엑셀</a>
                        <a href="pages/3_플랜트_분석.py?plant={row['플랜트']}&grade={row['등급']}&category={row['대분류']}" class='lot-download-btn' style='flex: 1;'>🔬분석</a>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

else:
    st.success("🎉 현재 감지된 주요 리스크가 없습니다. 안정적인 운영 상태입니다.")