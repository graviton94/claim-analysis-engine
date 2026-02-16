# Claim Analysis Engine

> **AI-Native Hybrid Intelligence Platform for Food Safety Quality Control**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.31.1-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-Internal-green.svg)]()

A hybrid intelligence platform that combines real-time operational monitoring (Track A) with ML-powered risk prediction (Track B) to enable early warning systems and data-driven resource allocation for Korean food manufacturers.

---

## 🎯 Quick Start

### Prerequisites
- Python 3.10 or higher
- 8GB+ RAM recommended
- 500MB+ disk space for data storage

### Installation

```bash
# Clone repository
git clone https://github.com/graviton94/claim-analysis-engine.git
cd claim-analysis-engine

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 🌟 Key Features

### ⚡ Track A: Operational Engine (Real-Time)
**Purpose:** Instant feedback for frontline managers
- **Real-time Risk Radar:** Automatic 🔴 Red (≥80) / 🟡 Yellow (≥50) / 🟢 Green (<50) classification
- **Action Cards:** One-click Excel export and drill-down navigation
- **Statistical Process Control:** Nelson Rules, CUSUM, sparse/dense anomaly detection
- **Performance:** <3 second latency for 4-month forecasts
- **Tech Stack:** NumPy, Statsmodels (lightweight only)

### 🧪 Track B: Simulation Lab (Deep Analysis)
**Purpose:** Precision forecasting for data analysts
- **Model Competition:** Prophet, SARIMAX, CatBoost, LSTM compete via backtesting
- **Dynamic Weighting:** Inverse MAE determines best ensemble combination
- **Sub-Category Allocation:** Time-weighted distribution with extinction detection
- **Performance:** 30-120 second latency (accuracy prioritized)
- **Tech Stack:** CatBoost, PyTorch, Optuna (heavy ML/DL allowed)

### 📊 Core Capabilities
- **54-Field Schema Management:** Immutable data integrity with Parquet partitioning
- **Hybrid Forecasting:** Run-rate + STL + LightGBM AutoML ensemble
- **Zero-Filling Logic:** Critical for "0 → spike" anomaly pattern detection
- **Multi-Page Dashboard:** Streamlit-based with Plotly visualizations

---

## 📂 Project Structure

```
claim-analysis-engine/
├── app.py                      # Main dashboard (Quality Control Tower)
├── core/                       # Business logic layer
│   ├── analytics.py            # Risk scoring engine (Nelson Rules, CUSUM)
│   ├── forecasting.py          # Track A: Lightweight ensemble
│   ├── storage.py              # Parquet I/O with PyArrow lazy loading
│   ├── etl.py                  # CSV/Excel ingestion pipeline
│   ├── config.py               # 54-field schema and constants
│   └── engine/                 # Track B: Heavy ML/DL
│       ├── trainer.py          # Model competition framework
│       ├── models.py           # Prophet, SARIMAX, CatBoost, LSTM
│       └── allocator.py        # Sub-category distribution logic
├── pages/                      # Streamlit multi-page app
│   ├── 1_데이터_업로드.py        # Data upload interface
│   ├── 2_매출수량_관리.py        # Sales volume management
│   ├── 3_플랜트_분석.py          # Deep diagnostic analysis
│   └── 4_예측_시뮬레이션.py      # ML forecasting lab
├── data/                       # Data storage (Git-ignored)
│   ├── hub/                    # Parquet partitioned claims (year/month)
│   ├── sales/                  # Sales volume data
│   └── models/                 # Trained ML models
├── docs/                       # Documentation
│   ├── README.md               # User guide
│   ├── project_master.md       # Technical master spec
│   └── ...
├── llms.txt                    # AI context map (file directory, tech stack)
├── ARCHITECTURE.md             # System architecture and design patterns
├── .github/copilot-instructions.md  # AI assistant guidelines
└── requirements.txt            # Python dependencies
```

---

## 🚀 Usage Guide

### 1. Upload Data
Navigate to **📤 데이터 업로드** page and upload CSV/Excel files with claim data. The system will:
- Validate 54-field schema
- Deduplicate by 상담번호 (case number)
- Partition by year/month in Parquet format

### 2. Monitor Real-Time Risks
The main **Quality Control Tower** dashboard displays:
- KPI metrics (total claims, risk distribution, trends)
- Risk Radar (high-risk plant identification)
- Trend charts with 4-month forecasts
- Color-coded severity indicators

### 3. Deep Dive Analysis
Use **📊 플랜트 분석** page for:
- Multi-dimensional pivot tables
- Lag analysis (제조-접수 time gap)
- Advanced risk scoring with diagnosis messages

### 4. Forecast Simulation
Use **🔮 예측 시뮬레이션** page for:
- ML model competition (Prophet vs SARIMAX vs AutoML)
- Weighted ensemble forecasting
- Sub-category level allocation
- Scenario validation

---

## 📖 Documentation

### For AI Agents
- **`llms.txt`** - Quick reference map (file directory, tech stack, critical constraints)
- **`ARCHITECTURE.md`** - Complete architecture (data flow, design patterns, algorithms)
- **`.github/copilot-instructions.md`** - Development guidelines with Karpathy principles

### For Developers
- **`docs/project_master.md`** - Technical master specification
- **`docs/refactor_spec.md`** - Refactoring requirements
- **`docs/milestone.md`** - Development roadmap

### For Users
- **`docs/README.md`** - System overview and user guide

---

## 🔧 Tech Stack

### Core Framework
- **Python 3.10+** - Main language (UTF-8-sig for Korean support)
- **Streamlit 1.31.1** - Multi-page dashboard framework
- **Pandas 2.1.4** - Data manipulation
- **PyArrow 14.0.1** - Parquet columnar storage (Hive partitioning)

### Track A (Operational)
- **NumPy 1.24.3** - Numerical computing
- **Statsmodels 0.14.0** - SARIMAX time-series
- **SciPy** - Statistical distributions (Poisson, Negative Binomial)

### Track B (Simulation)
- **CatBoost 1.2.1** - Gradient boosting
- **PyTorch 2.1.1** - LSTM deep learning
- **Optuna 3.14.0** - Hyperparameter optimization
- **Prophet** - Facebook's forecasting library
- **LightGBM** - Fast gradient boosting

### Visualization
- **Plotly** - Interactive charts
- **Matplotlib 3.8.2** - Statistical plotting

---

## ⚙️ Critical Constraints

### Data Integrity (Sacred Rules)
- ✅ **54-Field Schema:** System ONLY processes TARGET_54_COLS (immutable)
- ✅ **Partitioning:** Hive-style year=YYYY/month=MM/ (mandatory)
- ✅ **Zero-Filling:** Missing months filled with 0 (NOT NaN)
- ❌ **Never:** Train on current incomplete month (data leakage)

### Architecture Separation
- **Track A:** ONLY NumPy, Statsmodels allowed (forbidden: CatBoost, PyTorch, Optuna)
- **Track B:** All ML/DL libraries allowed
- **Rationale:** Track A must maintain <3s latency for real-time dashboard

### Risk Scoring (Fixed Thresholds)
- 🔴 **Red (Critical):** Score ≥ 80 (immediate attention)
- 🟡 **Yellow (Warning):** Score ≥ 50 (caution level)
- 🟢 **Green (Normal):** Score < 50 (safe)
- Special: Accident grade (사고) always scores 100

---

## 🎨 Design System

### Color Palette
```python
COLOR_RED = "#EF151E"      # Critical alerts
COLOR_YELLOW = "#FF9700"   # Warnings
COLOR_BLUE = "#006ECD"     # Information
COLOR_GRAY = "#2f3339"     # Neutral
```

### Filtering Modes
- **인입 (Inflow):** 사업부문 in ['식품', 'B2B식품'] AND 불만원인 not null
- **실적 (Performance):** + 불만원인 in ['제조불만', '구매불만', '고객불만족']
- **원본 (Raw):** No filtering
- **커스텀 (Custom):** User-selected criteria

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest test_analytics.py
pytest test_forecasting.py
pytest test_etl.py

# Run with coverage
pytest --cov=core --cov-report=html
```

---

## 📊 Performance

### Latency Targets
- **Track A (Dashboard):** <3 seconds for 4-month forecast
- **Track B (Simulation):** 30-120 seconds for model competition

### Memory Efficiency
- **Lazy Loading:** PyArrow Dataset reads only filtered partitions
- **Partitioning:** Year/month reduces scan size by ~95%
- **Compression:** Snappy codec achieves ~70% size reduction

---

## 🔒 Security & Privacy

- ✅ **Local Processing:** All ML training runs locally (no cloud APIs)
- ✅ **No Telemetry:** No external data transmission
- ✅ **Data Sanitization:** Null case numbers rejected at ETL
- ⚠️ **Network Security:** Relies on corporate network (no built-in auth)

---

## 🛠️ Development Guidelines

### Karpathy Principles (from `.github/copilot-instructions.md`)
1. **Minimalism:** No surprise refactoring, touch only surgical lines
2. **Simplicity:** Prefer functions over classes, flat logic over nesting
3. **YAGNI:** No future-proof flexibility or extra config options

### Code Quality Rules
- **Single Source of Truth:** All pages use `core.storage.load_and_filter_data()`
- **Session State Isolation:** Page 3 keys without suffix, Page 4 with `_sim` suffix
- **Error Handling:** Show warnings not crashes, graceful degradation

---

## 📝 License

Internal use only. Not licensed for external distribution.

---

## 🤝 Contributing

This is an internal project. For questions or issues:
1. Check `ARCHITECTURE.md` for design decisions
2. Review `llms.txt` for quick reference
3. Consult `.github/copilot-instructions.md` for development rules

---

## 📞 Support

For technical support or questions about the system:
- **Documentation:** See `docs/` directory
- **Architecture:** Read `ARCHITECTURE.md`
- **AI Context:** Refer to `llms.txt`

---

**Built with ❤️ for food safety excellence**
