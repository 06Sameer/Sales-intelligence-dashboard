# 📊 Sales Data Analysis Dashboard 2025

> **End-to-End Business Intelligence Project** — Python · Power BI · Machine Learning · DAX

---

## 🗂 Project Overview

A complete, portfolio-ready Business Intelligence solution that processes and analyzes **52,000+ sales records** spanning January 2024 – December 2025. The project delivers actionable insights on revenue growth, product performance, regional analysis, customer behaviour, and a 12-month revenue forecast.

| Metric | Value |
|---|---|
| Total Revenue | ₹116.86 Crore |
| Total Profit | ₹56.73 Crore |
| Profit Margin | 48.5% |
| Total Orders | 52,000 |
| Avg Order Value | ₹22,653 |
| Top Product | WorkStation Pro |
| Top Region | North India |
| Top Segment | Corporate |
| Forecast 2026 | ~₹59.3 Crore |

---

## 📁 Project Structure

```
sales_dashboard_project/
├── data/
│   ├── raw_sales_data.csv          ← 52,156 records (with noise)
│   ├── cleaned_sales_data.csv      ← 52,000 records, 32 columns
│   └── sales_forecast.csv          ← 12-month forecast output
│
├── scripts/
│   ├── 01_generate_dataset.py      ← Synthetic data generator
│   ├── 02_data_cleaning_eda.py     ← Cleaning + EDA + 10 charts
│   ├── 03_forecasting.py           ← ML models + forecast charts
│   └── 04_generate_report.js       ← Word report generator (Node.js)
│
├── dashboard/
│   └── DAX_Measures.dax            ← 30+ Power BI DAX measures
│
├── reports/
│   └── Business_Insights_Report.docx  ← 12-page professional report
│
├── visuals/                         ← 13 EDA + forecast charts (PNG)
│   ├── 01_kpi_summary.png
│   ├── 02_monthly_revenue_trend.png
│   ├── 03_regional_performance.png
│   ├── 04_category_performance.png
│   ├── 05_product_top_bottom10.png
│   ├── 06_customer_segmentation.png
│   ├── 07_payment_shipping.png
│   ├── 08_quarterly_revenue.png
│   ├── 09_discount_impact.png
│   ├── 10_correlation_heatmap.png
│   ├── 11_model_comparison.png
│   ├── 12_revenue_forecast_12months.png
│   └── 13_model_metrics.png
│
├── models/
│   └── model_metrics.json          ← MAE, RMSE, R² for all models
│
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ (for report generation only)

### Install Python Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### Install Node.js Dependencies (for report generation)

```bash
npm install -g docx
```

---

## 🚀 Run the Pipeline

Execute scripts in order:

```bash
# Step 1: Generate 52,000+ synthetic sales records
python scripts/01_generate_dataset.py

# Step 2: Clean data + run full EDA (generates 10 charts)
python scripts/02_data_cleaning_eda.py

# Step 3: Train ML models + generate 12-month forecast (3 charts)
python scripts/03_forecasting.py

# Step 4: Generate Word business insights report
node scripts/04_generate_report.js
```

---

## 📊 Dataset Schema

| Field | Type | Description |
|---|---|---|
| Order_ID | String | Unique order identifier |
| Order_Date | Date | Order placement date |
| Customer_ID | String | Customer identifier |
| Customer_Segment | Category | Corporate / SME / Retail / Government / Startup |
| Product_Category | Category | Electronics / Furniture / Office Supplies / Clothing / Sports |
| Region | Category | North / South / East / West |
| Revenue | Decimal | Net revenue after discount (INR) |
| Profit | Decimal | Revenue minus cost (INR) |
| Profit_Margin | Decimal | Profit as % of Revenue |

Full schema documented in `Business_Insights_Report.docx` Section 11.

---

## 🤖 Machine Learning Models

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ₹2.55M | ₹3.04M | -2.78 |
| Ridge Regression | ₹3.32M | ₹3.60M | -4.30 |
| **Random Forest** ✅ | **₹1.71M** | **₹2.13M** | **-0.86** |
| Gradient Boosting | ₹3.48M | ₹3.77M | -4.84 |

**Best Model: Random Forest** — lowest MAE and RMSE on 24-month time series.

---

## 📋 Power BI Dashboard Pages

| Page | Focus Area |
|---|---|
| 1. Executive Overview | KPI cards, revenue trends, region & category |
| 2. Product Analytics | Top/bottom products, profitability |
| 3. Regional Performance | Map, state drill-down |
| 4. Customer Insights | Segments, AOV, payment methods |
| 5. Sales Forecasting | 6 & 12-month forecast with confidence bands |

**Setup:** Import `cleaned_sales_data.csv` → Add `DAX_Measures.dax` measures → Build 5-page layout.

---

## 💡 Key Business Insights

1. **North India** is the top-performing region (32% revenue share) — maintain current investment.
2. **East India** is significantly underpenetrated — high-priority expansion opportunity.
3. **Corporate segment** drives highest AOV with lowest discount sensitivity.
4. **Q4 (Oct–Dec)** generates 30–35% more revenue than off-peak quarters — stock up accordingly.
5. **Discounts above 10%** compress margins with limited volume benefit outside festival seasons.
6. **UPI (30%)** is the dominant payment method — ensure frictionless UPI checkout.
7. **WorkStation Pro & iPhone 15** are consistent revenue champions — protect availability.

---

## 📄 Resume Description

> Built an end-to-end Business Intelligence solution analyzing 52,000+ sales transactions across 24 months. Developed Python data pipeline (pandas, scikit-learn) for automated data cleaning, EDA, and ML-based sales forecasting (Random Forest). Designed a 5-page interactive Power BI dashboard with 30+ DAX measures, drill-through filters, and KPI tracking. Delivered actionable insights on regional performance, customer segmentation, and product profitability to support executive decision-making.

---

## 🔧 Technologies Used

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `Power BI` · `DAX` · `Node.js` · `docx`

---

*Prepared by: Data Analytics Team | Classification: Portfolio / Public*
