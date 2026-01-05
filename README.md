# Claim Data Intelligence System

AI-powered Claim Data Intelligence System: Automated ETL (CSV/XLSX/JSON), Dynamic Streamlit Dashboard, and Proactive Risk Forecasting using ML & Seasonal Analysis.

## 🌟 Features

### 1. ETL Pipeline
- **Multi-format Support**: Ingest data from CSV, XLSX, and JSON files
- **Data Standardization**: Automatic schema normalization and data type conversion
- **Parquet Database**: High-performance columnar storage for efficient querying
- **Batch Processing**: Process multiple files and consolidate into unified datasets

### 2. Interactive Dashboard
- **Real-time Visualization**: Dynamic charts and graphs for claim trends
- **Advanced Filtering**: Filter by date range, claim type, status, and more
- **Key Metrics**: Total claims, amounts, averages, and distributions
- **Multiple Views**: Time series, pie charts, histograms, and data tables

### 3. Risk Forecasting
- **ML-based Prediction**: Forecast future claim trends using historical data
- **Seasonality Detection**: Automatic identification of seasonal patterns
- **Risk Assessment**: Proactive warnings with risk scores (Low/Medium/High)
- **Confidence Intervals**: Statistical bounds for forecast reliability

## 🏗️ Architecture

```
claim-analysis-engine/
├── app.py                 # Streamlit entry point
├── core/                  # Processing & ML engines
│   ├── __init__.py
│   ├── etl.py            # ETL processor
│   ├── storage.py        # Parquet storage manager
│   └── forecasting.py    # ML forecasting engine
├── data/                  # Parquet storage (gitignored)
├── indices/              # Master data
│   ├── claim_types.csv
│   └── claim_status.csv
└── requirements.txt
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/graviton94/claim-analysis-engine.git
cd claim-analysis-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Starting the Dashboard

Run the Streamlit application:
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Using the System

#### 1. ETL & Data Ingestion
- Navigate to **"📥 ETL & Data Ingestion"** in the sidebar
- Upload your CSV, XLSX, or JSON files
- Click **"Process Files"** to convert to Parquet format
- Or use **"Generate Sample Data"** to create test data

#### 2. Analytics Dashboard
- Navigate to **"📈 Analytics Dashboard"**
- Use sidebar filters to refine your view:
  - Date range
  - Claim types
  - Status
- Explore visualizations:
  - Claims over time (dual-axis chart)
  - Distribution by type (pie chart)
  - Status breakdown (bar chart)
  - Amount distribution (histogram)

#### 3. Risk Forecasting
- Navigate to **"🔮 Risk Forecasting"**
- Set forecast period (7-90 days)
- Click **"Generate Forecast"**
- Review:
  - Risk assessment (level and score)
  - Seasonality analysis
  - Forecast visualization with confidence intervals
  - Detailed forecast data table

## 📊 Data Format

### Input Data Requirements

Your claim data should include the following columns (column names are flexible):

- **Date column**: Any column with "date" in the name (e.g., `claim_date`, `date_filed`)
- **Amount column**: Any column with "amount" in the name (e.g., `claim_amount`, `total_amount`)
- **Type/Category column**: Any column with "type" or "category" (e.g., `claim_type`, `category`)
- **Status column**: Any column with "status" (e.g., `status`, `claim_status`)

### Example CSV Format

```csv
claim_id,claim_date,claim_amount,claim_type,status
CLM000001,2024-01-15,15000.50,Medical,APPROVED
CLM000002,2024-01-16,3500.00,Auto,PENDING
CLM000003,2024-01-17,8200.75,Property,PAID
```

### Supported Claim Types

The system includes predefined claim types in `indices/claim_types.csv`:
- Medical (High risk)
- Auto (Medium risk)
- Property (Medium risk)
- Life (High risk)
- Travel (Low risk)
- Disability (High risk)
- Dental (Low risk)
- Vision (Low risk)

### Supported Statuses

Defined in `indices/claim_status.csv`:
- PENDING - Claim is under review
- APPROVED - Claim has been approved
- REJECTED - Claim has been rejected
- PAID - Payment processed
- APPEALED - Under appeal
- CANCELLED - Claim cancelled

## 🔧 Technical Details

### Core Modules

#### ETLProcessor (`core/etl.py`)
- Reads CSV/XLSX/JSON files
- Standardizes data schema
- Converts to Parquet format
- Supports batch processing

#### ParquetStorage (`core/storage.py`)
- High-performance data loading
- Advanced querying with filters
- Statistical summaries
- File management

#### RiskForecaster (`core/forecasting.py`)
- Time series preparation
- Seasonality detection (7-day and 30-day cycles)
- Simple moving average forecasting
- Risk score calculation
- Confidence interval estimation

### Dependencies

- **streamlit**: Interactive dashboard framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **pyarrow**: Parquet file format support
- **openpyxl**: Excel file reading
- **scikit-learn**: Machine learning utilities
- **plotly**: Interactive visualizations

## 🎯 Use Cases

1. **Insurance Companies**: Monitor claim trends and predict future liabilities
2. **Healthcare Providers**: Analyze medical claim patterns and forecast costs
3. **Risk Managers**: Identify high-risk periods and take proactive measures
4. **Data Analysts**: Explore claim data with interactive visualizations
5. **Finance Teams**: Budget planning based on claim forecasts

## 📈 Performance

- **Parquet Format**: 10x faster than CSV for large datasets
- **Columnar Storage**: Efficient compression and query performance
- **Real-time Filtering**: Interactive dashboard with minimal latency
- **Scalable**: Handles millions of records efficiently

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For questions or issues, please open an issue on GitHub.
