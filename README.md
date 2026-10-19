# India Power Analytics - Electricity Demand Forecasting System

An advanced, AI-powered Electricity Demand Forecasting System upgraded from a historical analytics dashboard. The platform analyzes historical electricity generation data (2013-2023) across India and predicts demand for future months and years using Machine Learning and Time Series Forecasting models.

The system is designed with a modern, glassmorphic UI matching a government-grade electricity analytics platform and is structured with a modular, scalable architecture.

---

## 🌟 Key Features

### 1. Future Demand Prediction Page (`/prediction`)
*   **Targeted Projections**: Select State/Region, Power Source, and Forecast Period (Next Month, Next 3/6 Months, Next Year, or Custom date picker).
*   **Predictive Indicators**: Real-time calculated counters showing Predicted Demand (MU), Model Confidence (%), Expected Growth (%), and Grid Trend direction.
*   **Interactive Visualizations**: High-fidelity Plotly charts including:
    *   *Timeline & Bounds*: Historical and future forecast lines with shaded 95% Confidence Intervals.
    *   *Yearly Compare*: Year-over-year energy consumption totals.
    *   *Monthly Seasonality*: Statistical month-by-month demand averages.
    *   *State Rankings*: Ranks the selected state against top energy-consuming states in India.
    *   *Source Shares*: Donut share breakdown of current power generation sources.
*   **AI-Generated Insights**: Computes data-driven observations on load volatility, baseline trends, and renewable growth.
*   **Prediction History Sidebar**: Displays user forecast history. Clicking any previous run instantly restores parameters and re-renders visual graphs.

### 2. Model Performance Dashboard (`/model-performance`)
*   **Validation Grid**: Compares test-set metrics (Accuracy, MAE, MSE, RMSE, MAPE) and speeds (Training/Prediction times) across all models.
*   **Best Model Highlight**: Automatically evaluates and highlights the best-performing model with a specialized success-card banner.
*   **Interactive Bar Chart**: Embeds Plotly comparative accuracy figures.

### 3. Comprehensive Export Utility
*   **Professional PDF Reports**: Generates stylized, publication-ready reports using ReportLab containing summary metadata, key metrics cards, embedded matplotlib forecast charts, and validation comparison tables.
*   **CSV Data Export**: Compiles and downloads forecast timeline values instantly.
*   **PNG Visuals**: Download individual interactive charts straight from graph headers.

### 4. Self-Healing Database Abstraction
*   **PostgreSQL Support**: Connects to the local PostgreSQL database (`power_analytics_db`) by default.
*   **SQLite Fallback**: If PostgreSQL is offline, the app transparently falls back to a local SQLite database (`instance/power_analytics.db`), auto-creating all schemas (`users`, `user_history`, `prediction_history`) on the fly. No manual database setup is required.

---

## 📂 Codebase Architecture

The project has been refactored into a clean, modular structure:

```text
C:\Users\hpwor\Documents\Electricity-Demand-Projection-System--main\
│
├── app.py                              # Core Flask application loader
├── Daily_Power_Gen_Source_march_23.csv # Daily source generation dataset (2013-2023)
├── Daily_Power_Gen_States_march_23.csv # Daily state energy met dataset (2013-2023)
├── requirements.txt                    # Project dependencies list
├── README.md                           # Developer documentation
│
├── models/
│   ├── __init__.py
│   └── user.py                         # Flask-Login User model class
│
├── utils/
│   ├── __init__.py
│   ├── db.py                           # Self-healing database wrapper (Postgres + SQLite)
│   └── charts.py                       # Plotly JSON figure generators
│
├── services/
│   ├── __init__.py
│   ├── data_service.py                 # Time-series aggregation & downscaling
│   ├── ml_service.py                   # ML training and multi-step forecast execution
│   ├── insights_service.py             # Rule-based AI insight engine
│   └── report_service.py               # PDF document builder (ReportLab + Matplotlib)
│
├── routes/
│   ├── __init__.py
│   └── prediction.py                   # Forecaster blueprints (UI pages & REST APIs)
│
├── templates/
│   ├── index.html                      # Existing dashboard page (navbar updated)
│   ├── login.html                      # User login page
│   ├── register.html                   # User registration page
│   ├── prediction.html                 # Future Demand Prediction interface
│   └── model_performance.html          # ML validation dashboard
│
└── instance/
    └── power_analytics.db              # Local SQLite database file (fallback storage)
```

---

## 🤖 Machine Learning Models

When a forecast is requested, the system isolates the historical monthly target, downscales regional source generation using monthly state share coefficients, and trains four forecasting engines:
1.  **Linear Regression**: Fits lag-based variables (`lag_1`, `lag_2`, `lag_12`) and chronological trend parameters.
2.  **Random Forest Regressor**: Assembles decision tree estimators with restricted max-depth to prevent overfitting on shorter monthly series.
3.  **Support Vector Regressor (SVR)**: Trains Radial Basis Function kernel parameters on standard-scaled training matrices.
4.  **Holt-Winters Exponential Smoothing**: Applies triple exponential smoothing using explicit parameter sets (`optimized=False`) to guarantee instant, deadlock-free convergence on sparse series.

The model with the highest test-set accuracy (`100 - MAPE`) is selected to generate the final recursive multi-step projections with expanding 95% Confidence Intervals ($1.96 \times \text{RMSE} \times \sqrt{t}$).

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```
*Note: Key packages include `flask`, `flask-login`, `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `plotly`, `matplotlib`, and `reportlab`.*

### 3. Run the Application
Start the Flask server:
```bash
python app.py
```

On startup, the system will output database initialization logs:
*   If your local PostgreSQL service is running with matching credentials, it will connect.
*   Otherwise, it will print `PostgreSQL connection failed. Falling back to SQLite.` and automatically boot using the local `instance/power_analytics.db` SQLite file.

### 4. Open in Browser
Open your browser and navigate to:
```text
http://127.0.0.1:5000/
```
https://electricity-demand-projection-system.onrender.com/

If you are using the clean SQLite database fallback:
*   Click **Register** to create a test user (e.g., username `demo`, password `12345678`).
*   You will be logged in automatically and can access the forecast features.
