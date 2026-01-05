"""
Claim Data Intelligence System - Interactive Dashboard
Streamlit-based UI for claim trend visualization, ETL, and risk forecasting
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path

from core import ETLProcessor, ParquetStorage, RiskForecaster


# Page configuration
st.set_page_config(
    page_title="Claim Data Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def init_components():
    """Initialize ETL, Storage, and Forecasting components"""
    etl = ETLProcessor(data_dir="data")
    storage = ParquetStorage(data_dir="data")
    forecaster = RiskForecaster()
    return etl, storage, forecaster

etl, storage, forecaster = init_components()


def main():
    """Main application"""
    
    st.title("📊 Claim Data Intelligence System")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        ["📥 ETL & Data Ingestion", "📈 Analytics Dashboard", "🔮 Risk Forecasting"]
    )
    
    if page == "📥 ETL & Data Ingestion":
        show_etl_page()
    elif page == "📈 Analytics Dashboard":
        show_analytics_page()
    elif page == "🔮 Risk Forecasting":
        show_forecasting_page()


def show_etl_page():
    """ETL and Data Ingestion page"""
    
    st.header("📥 ETL & Data Ingestion")
    st.markdown("Upload and process CSV, XLSX, or JSON files into high-performance Parquet database")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload claim data files",
        type=['csv', 'xlsx', 'xls', 'json'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.subheader("Uploaded Files")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size} bytes)")
        
        if st.button("Process Files", type="primary"):
            with st.spinner("Processing files..."):
                try:
                    # Save uploaded files temporarily
                    temp_dir = Path("/tmp/uploads")
                    temp_dir.mkdir(exist_ok=True)
                    
                    file_paths = []
                    for file in uploaded_files:
                        temp_path = temp_dir / file.name
                        with open(temp_path, 'wb') as f:
                            f.write(file.getbuffer())
                        file_paths.append(str(temp_path))
                    
                    # Process files
                    if len(file_paths) == 1:
                        output_path = etl.process_file(file_paths[0])
                    else:
                        output_path = etl.process_multiple_files(file_paths)
                    
                    st.success(f"✅ Successfully processed and saved to: {output_path}")
                    
                    # Show preview
                    df = storage.load_data()
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10))
                    
                    # Show statistics
                    st.subheader("Data Statistics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Records", len(df))
                    col2.metric("Total Columns", len(df.columns))
                    col3.metric("Memory Usage (MB)", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing files: {str(e)}")
    
    # Show existing data
    st.markdown("---")
    st.subheader("Existing Data Files")
    
    files = storage.list_files()
    if files:
        st.write(f"Found {len(files)} Parquet file(s):")
        for file in files:
            st.write(f"- {file}")
    else:
        st.info("No data files found. Upload files above to get started.")
    
    # Sample data generator
    st.markdown("---")
    st.subheader("Generate Sample Data")
    
    if st.button("Generate Sample Claim Data"):
        with st.spinner("Generating sample data..."):
            df = generate_sample_data(1000)
            output_path = etl.save_to_parquet(df, "sample_claims")
            st.success(f"✅ Generated 1000 sample claims and saved to: {output_path}")
            st.dataframe(df.head(10))


def show_analytics_page():
    """Analytics Dashboard page"""
    
    st.header("📈 Analytics Dashboard")
    st.markdown("Interactive visualization of claim trends and patterns")
    
    # Load data
    df = storage.load_data()
    
    if df.empty:
        st.warning("⚠️ No data available. Please upload data in the ETL module or generate sample data.")
        return
    
    # Detect date and amount columns
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    amount_cols = [col for col in df.columns if 'amount' in col.lower()]
    type_cols = [col for col in df.columns if 'type' in col.lower() or 'category' in col.lower()]
    status_cols = [col for col in df.columns if 'status' in col.lower()]
    
    date_col = date_cols[0] if date_cols else None
    amount_col = amount_cols[0] if amount_cols else None
    type_col = type_cols[0] if type_cols else None
    status_col = status_cols[0] if status_cols else None
    
    # Filters
    st.sidebar.subheader("Filters")
    
    # Date range filter
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            df = df[(df[date_col] >= pd.Timestamp(date_range[0])) & 
                   (df[date_col] <= pd.Timestamp(date_range[1]))]
    
    # Type filter
    if type_col:
        unique_types = df[type_col].unique().tolist()
        selected_types = st.sidebar.multiselect(
            "Claim Types",
            options=unique_types,
            default=unique_types
        )
        if selected_types:
            df = df[df[type_col].isin(selected_types)]
    
    # Status filter
    if status_col:
        unique_statuses = df[status_col].unique().tolist()
        selected_statuses = st.sidebar.multiselect(
            "Status",
            options=unique_statuses,
            default=unique_statuses
        )
        if selected_statuses:
            df = df[df[status_col].isin(selected_statuses)]
    
    # Key Metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Claims", f"{len(df):,}")
    
    with col2:
        if amount_col:
            total_amount = df[amount_col].sum()
            st.metric("Total Amount", f"${total_amount:,.2f}")
    
    with col3:
        if amount_col:
            avg_amount = df[amount_col].mean()
            st.metric("Average Amount", f"${avg_amount:,.2f}")
    
    with col4:
        if date_col:
            date_span = (df[date_col].max() - df[date_col].min()).days
            st.metric("Date Range (days)", f"{date_span}")
    
    # Visualizations
    st.markdown("---")
    
    # Time series chart
    if date_col and amount_col:
        st.subheader("Claims Over Time")
        
        # Aggregate by date
        daily_data = df.groupby(df[date_col].dt.date).agg({
            amount_col: ['sum', 'count']
        }).reset_index()
        daily_data.columns = ['date', 'total_amount', 'claim_count']
        
        # Create dual-axis chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data['total_amount'],
            name='Total Amount',
            yaxis='y',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.add_trace(go.Bar(
            x=daily_data['date'],
            y=daily_data['claim_count'],
            name='Claim Count',
            yaxis='y2',
            opacity=0.5,
            marker=dict(color='#ff7f0e')
        ))
        
        fig.update_layout(
            xaxis=dict(title='Date'),
            yaxis=dict(title='Total Amount ($)', side='left'),
            yaxis2=dict(title='Claim Count', side='right', overlaying='y'),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        if type_col and amount_col:
            st.subheader("Claims by Type")
            type_data = df.groupby(type_col)[amount_col].agg(['sum', 'count']).reset_index()
            type_data.columns = ['type', 'total_amount', 'count']
            
            fig = px.pie(
                type_data,
                values='total_amount',
                names='type',
                title='Total Amount by Type'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if status_col:
            st.subheader("Claims by Status")
            status_data = df[status_col].value_counts().reset_index()
            status_data.columns = ['status', 'count']
            
            fig = px.bar(
                status_data,
                x='status',
                y='count',
                title='Claim Count by Status',
                color='count',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Amount distribution
    if amount_col:
        st.subheader("Amount Distribution")
        fig = px.histogram(
            df,
            x=amount_col,
            nbins=50,
            title='Distribution of Claim Amounts',
            labels={amount_col: 'Claim Amount ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("---")
    st.subheader("Filtered Data")
    st.dataframe(df, use_container_width=True)


def show_forecasting_page():
    """Risk Forecasting page"""
    
    st.header("🔮 Risk Forecasting")
    st.markdown("ML-based prediction using seasonality and historical trends")
    
    # Load data
    df = storage.load_data()
    
    if df.empty:
        st.warning("⚠️ No data available. Please upload data in the ETL module or generate sample data.")
        return
    
    # Detect date and amount columns
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    amount_cols = [col for col in df.columns if 'amount' in col.lower()]
    
    if not date_cols or not amount_cols:
        st.error("❌ Data must contain date and amount columns for forecasting.")
        return
    
    date_col = date_cols[0]
    amount_col = amount_cols[0]
    
    # Forecasting parameters
    st.sidebar.subheader("Forecast Settings")
    forecast_days = st.sidebar.slider("Forecast Period (days)", 7, 90, 30)
    
    # Run forecast
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                forecast, seasonality, risk = forecaster.forecast_claims(
                    df,
                    date_col,
                    amount_col,
                    forecast_days
                )
                
                # Display risk assessment
                st.subheader("Risk Assessment")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_color = {
                        "Low": "🟢",
                        "Medium": "🟡",
                        "High": "🔴"
                    }
                    st.metric(
                        "Risk Level",
                        f"{risk_color.get(risk['risk_level'], '⚪')} {risk['risk_level']}"
                    )
                
                with col2:
                    st.metric("Risk Score", f"{risk['risk_score']}/100")
                
                with col3:
                    st.metric(
                        "Forecasted Avg",
                        f"${risk['avg_forecast']:,.2f}",
                        delta=f"{((risk['avg_forecast'] / risk['historical_avg'] - 1) * 100):.1f}%"
                    )
                
                # Risk details
                st.markdown("---")
                st.subheader("Risk Details")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Historical Average:**", f"${risk['historical_avg']:,.2f}")
                    st.write("**Forecasted Average:**", f"${risk['avg_forecast']:,.2f}")
                
                with col2:
                    st.write("**Forecasted Maximum:**", f"${risk['max_forecast']:,.2f}")
                    st.write("**Risk Threshold:**", f"${risk['threshold']:,.2f}")
                
                # Seasonality information
                st.markdown("---")
                st.subheader("Seasonality Analysis")
                
                if seasonality['has_seasonality']:
                    st.success(f"✅ Seasonal pattern detected with period of {seasonality['period']} days (strength: {seasonality['strength']:.2f})")
                else:
                    st.info("ℹ️ No strong seasonal pattern detected in the data")
                
                if seasonality['patterns']:
                    st.write("**Pattern Strengths:**")
                    for period, strength in seasonality['patterns'].items():
                        st.write(f"- {period}-day cycle: {strength:.3f}")
                
                # Forecast visualization
                st.markdown("---")
                st.subheader("Forecast Visualization")
                
                # Prepare historical data
                df_copy = df.copy()
                df_copy[date_col] = pd.to_datetime(df_copy[date_col])
                historical = df_copy.groupby(pd.Grouper(key=date_col, freq='D'))[amount_col].sum().reset_index()
                historical.columns = ['ds', 'y']
                
                # Create forecast chart
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical['ds'],
                    y=historical['y'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True
                ))
                
                fig.update_layout(
                    title='Claims Forecast with Confidence Intervals',
                    xaxis_title='Date',
                    yaxis_title='Total Claim Amount ($)',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast data table
                st.markdown("---")
                st.subheader("Forecast Data")
                
                forecast_display = forecast.copy()
                forecast_display['ds'] = forecast_display['ds'].dt.date
                forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                
                st.dataframe(
                    forecast_display.style.format({
                        'Forecast': '${:,.2f}',
                        'Lower Bound': '${:,.2f}',
                        'Upper Bound': '${:,.2f}'
                    }),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error generating forecast: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def generate_sample_data(num_records=1000):
    """Generate sample claim data for testing"""
    import random
    from datetime import timedelta
    
    claim_types = ['Medical', 'Auto', 'Property', 'Life', 'Travel', 'Disability', 'Dental', 'Vision']
    statuses = ['PENDING', 'APPROVED', 'REJECTED', 'PAID', 'APPEALED', 'CANCELLED']
    
    start_date = datetime.now() - timedelta(days=365)
    
    data = []
    for i in range(num_records):
        claim_date = start_date + timedelta(days=random.randint(0, 365))
        claim_type = random.choice(claim_types)
        
        # Amount varies by type
        if claim_type in ['Medical', 'Life']:
            amount = random.uniform(5000, 50000)
        elif claim_type in ['Auto', 'Property']:
            amount = random.uniform(2000, 25000)
        else:
            amount = random.uniform(100, 5000)
        
        data.append({
            'claim_id': f'CLM{i+1:06d}',
            'claim_date': claim_date,
            'claim_amount': round(amount, 2),
            'claim_type': claim_type,
            'status': random.choice(statuses)
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()
