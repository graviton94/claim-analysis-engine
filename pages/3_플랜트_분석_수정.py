    # --- 시각화 ---
    st.subheader(f"📈 분석 결과 ({grade_mode} / {search_mode})")
    
    # --- 1. 그래프 먼저 생성 (graph_index 기준) ---
    try:
        # 2개년 분리
        end_year = pd.to_datetime(end_date).year
        recent_years_set = {end_year, end_year - 1}
        recent_months = [c for c in all_months_in_range if int(c[:4]) in recent_years_set]
        
        # graph_index 기준으로 피벗
        pivot_for_graph = pd.pivot_table(
            filtered_df_step3,
            index=graph_index,
            columns='접수월_str',
            values='상담번호',
            aggfunc='count',
            fill_value=0
        )
        pivot_for_graph = pivot_for_graph.reindex(columns=all_months_in_range, fill_value=0)
        
        # 그래프 제목 계산
        start_month = recent_months[0] if recent_months else all_months_in_range[0]
        end_month = recent_months[-1] if recent_months else all_months_in_range[-1]
        st.markdown(f"#### 📊 2개년 추이 분석 (그래프 상 시작 {start_month} ~ 끝 {end_month})")
        
        # Plotly 선 그래프 구성
        fig = px.line(title=f"2개년 클레임 건수 추이 ({graph_index} 기준)")
        
        # 색상 팔레트
        colors = px.colors.qualitative.Plotly
        
        # 각 graph_index 값별 선 그리기
        for idx, category in enumerate(pivot_for_graph.index):
            color = colors[idx % len(colors)]
            category_data = pivot_for_graph.loc[category]
            
            # Recent 데이터: 월별 실제 값
            if recent_months:
                recent_data = category_data[recent_months]
                fig.add_scatter(
                    x=recent_months,
                    y=recent_data.values,
                    mode='lines+markers',
                    name=f'{category}',
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    legendgroup=category,
                    showlegend=True
                )
        
        fig.update_layout(
            xaxis_title="월별 (Month)",
            yaxis_title="클레임 건수 (건)",
            hovermode='x unified',
            height=450,
            template="plotly_white",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"그래프 생성 중 오류: {e}")
    
    st.divider()
    
    # --- 2. 피벗 테이블 (pivot_indices 기준) ---
    tab1, tab2, tab3 = st.tabs(["피벗 테이블", "Lag 분석", "원본 데이터"])
