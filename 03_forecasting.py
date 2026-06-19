"""
============================================================
  Sales Data Analysis Dashboard 2025
  Script 03: Sales Forecasting — ML Models
  Author: Data Analytics Team
  Description: Linear Regression, Random Forest forecasting
               with MAE / RMSE / R² evaluation. Forecasts
               next 6 and 12 months revenue.
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings, os, json

warnings.filterwarnings("ignore")

BASE    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(BASE, "data")
VISUALS = os.path.join(BASE, "visuals")
MODELS  = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

DARK_BLUE = "#1A3C6E"
ACCENT    = "#2E86AB"
RED       = "#E63946"
GREEN     = "#2EC4B6"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F8F9FA",
    "axes.grid": True, "grid.color": "#E9ECEF", "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold",
})

def save_fig(name):
    path = os.path.join(VISUALS, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   📊 Saved: visuals/{name}.png")

# ── Load cleaned data ─────────────────────────────────────
print("\n" + "=" * 60)
print("  Sales Forecasting — ML Pipeline")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA, "cleaned_sales_data.csv"), parse_dates=["Order_Date"])

# ── Monthly aggregation ───────────────────────────────────
monthly = (df.groupby("Year_Month")
             .agg(Revenue=("Revenue","sum"),
                  Orders=("Order_ID","nunique"),
                  Avg_Price=("Unit_Price","mean"),
                  Total_Qty=("Quantity","sum"),
                  Avg_Discount=("Discount","mean"))
             .reset_index()
             .sort_values("Year_Month"))

monthly["Period_Idx"] = range(len(monthly))
monthly["Year_Month_dt"] = pd.to_datetime(monthly["Year_Month"])
monthly["Month"]    = monthly["Year_Month_dt"].dt.month
monthly["Quarter"]  = monthly["Year_Month_dt"].dt.quarter
monthly["Month_Sin"] = np.sin(2 * np.pi * monthly["Month"] / 12)
monthly["Month_Cos"] = np.cos(2 * np.pi * monthly["Month"] / 12)
monthly["Lag1"]     = monthly["Revenue"].shift(1)
monthly["Lag3"]     = monthly["Revenue"].shift(3)
monthly["Lag12"]    = monthly["Revenue"].shift(12)
monthly["Rolling3"] = monthly["Revenue"].shift(1).rolling(3).mean()
monthly = monthly.dropna().reset_index(drop=True)

FEATURES = ["Period_Idx","Month","Quarter","Month_Sin","Month_Cos",
            "Orders","Avg_Price","Total_Qty","Avg_Discount",
            "Lag1","Lag3","Rolling3"]

X = monthly[FEATURES].values
y = monthly["Revenue"].values

# ── Train / Test split (last 4 months = test) ─────────────
split = len(X) - 4
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Model Zoo ─────────────────────────────────────────────
models = {
    "Linear Regression":     LinearRegression(),
    "Ridge Regression":      Ridge(alpha=10),
    "Random Forest":         RandomForestRegressor(n_estimators=200, max_depth=8,
                                                   min_samples_leaf=2, random_state=42),
    "Gradient Boosting":     GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                        max_depth=4, random_state=42),
}

results = {}
print("\n  Model Evaluation")
print(f"  {'Model':<25} {'MAE':>12} {'RMSE':>12} {'R²':>8}")
print("  " + "-" * 60)

for name, model in models.items():
    if "Regression" in name:
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "model": model, "pred": pred}
    print(f"  {name:<25} ₹{mae/1e6:>8.2f}M  ₹{rmse/1e6:>8.2f}M  {r2:>6.3f}")

# Best model by RMSE
best_name = min(results, key=lambda k: results[k]["RMSE"])
best      = results[best_name]
print(f"\n  🏆 Best Model: {best_name}  (RMSE ₹{best['RMSE']/1e6:.2f}M, R²={best['R2']:.3f})")

# ── Future Forecast ───────────────────────────────────────
last_month = monthly["Year_Month_dt"].max()
last_idx   = monthly["Period_Idx"].max()
last_rev   = monthly["Revenue"].values
last_orders = monthly["Orders"].mean()
last_price  = monthly["Avg_Price"].mean()
last_qty    = monthly["Total_Qty"].mean()
last_disc   = monthly["Avg_Discount"].mean()

future_rows = []
rolling_rev = list(last_rev[-3:])
lag1 = last_rev[-1]
lag3 = last_rev[-3]
rolling3 = np.mean(rolling_rev)

for i in range(1, 13):
    future_date = last_month + pd.DateOffset(months=i)
    m = future_date.month
    q = future_date.quarter
    idx = last_idx + i
    row = [idx, m, q, np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12),
           last_orders, last_price, last_qty, last_disc, lag1, lag3, rolling3]
    future_rows.append(row)

X_future = np.array(future_rows)
if "Regression" in best_name:
    X_future_s = scaler.transform(X_future)
    forecast   = best["model"].predict(X_future_s)
else:
    forecast   = best["model"].predict(X_future)

forecast = np.maximum(forecast, 0)
future_dates = [last_month + pd.DateOffset(months=i) for i in range(1, 13)]

forecast_df = pd.DataFrame({
    "Month": [d.strftime("%b %Y") for d in future_dates],
    "Forecast_Revenue": forecast.round(0),
    "Lower_Bound":      (forecast * 0.90).round(0),
    "Upper_Bound":      (forecast * 1.10).round(0),
})

print(f"\n  Revenue Forecast — Next 12 Months ({best_name})")
print(f"  {'Month':<12} {'Forecast (₹M)':>14} {'Low':>10} {'High':>10}")
print("  " + "-" * 48)
for _, row in forecast_df.iterrows():
    print(f"  {row['Month']:<12} ₹{row['Forecast_Revenue']/1e6:>10.2f}M  "
          f"₹{row['Lower_Bound']/1e6:>6.1f}M  ₹{row['Upper_Bound']/1e6:>6.1f}M")

forecast_df.to_csv(os.path.join(DATA, "sales_forecast.csv"), index=False)

# ── Chart 1: Actual vs Predicted ─────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
dates_all = monthly["Year_Month_dt"].values
ax.plot(dates_all, monthly["Revenue"]/1e6, color=DARK_BLUE, linewidth=2.5,
        label="Actual Revenue", marker="o", markersize=4)

test_dates = monthly["Year_Month_dt"].values[split:]
for mname, res in results.items():
    alpha = 0.9 if mname == best_name else 0.4
    lw    = 2.2 if mname == best_name else 1.2
    ax.plot(test_dates, res["pred"]/1e6, linewidth=lw, alpha=alpha,
            label=f"{mname} {'★' if mname==best_name else ''}")

ax.set_title("Model Comparison — Actual vs Predicted Revenue (Test Period)")
ax.set_xlabel("Month"); ax.set_ylabel("Revenue (₹ Millions)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}M"))
ax.legend(fontsize=8)
ax.axvline(x=test_dates[0], color="gray", linestyle="--", alpha=0.7, label="Train/Test Split")
save_fig("11_model_comparison")

# ── Chart 2: 12-Month Forecast ────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(monthly["Year_Month_dt"], monthly["Revenue"]/1e6,
        color=DARK_BLUE, linewidth=2.5, marker="o", markersize=4, label="Historical Revenue")
ax.plot(future_dates, forecast/1e6, color=RED, linewidth=2.5,
        marker="D", markersize=6, linestyle="--", label=f"Forecast ({best_name})")
ax.fill_between(future_dates,
                forecast_df["Lower_Bound"]/1e6,
                forecast_df["Upper_Bound"]/1e6,
                alpha=0.18, color=RED, label="90% Confidence Band")
ax.axvline(x=future_dates[0], color="gray", linestyle=":", alpha=0.7)
ax.text(future_dates[0], ax.get_ylim()[1]*0.97, "← Historical | Forecast →",
        ha="center", fontsize=8.5, color="gray")
ax.set_title(f"12-Month Revenue Forecast — {best_name}")
ax.set_ylabel("Revenue (₹ Millions)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}M"))
ax.legend()
save_fig("12_revenue_forecast_12months")

# ── Chart 3: Model Metrics Bar ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold", color=DARK_BLUE)
colors = [GREEN if k==best_name else "#ADB5BD" for k in results]

axes[0].bar(results.keys(), [v["MAE"]/1e6 for v in results.values()], color=colors, edgecolor="white")
axes[0].set_title("MAE (₹ Millions)"); axes[0].tick_params(axis="x", rotation=20)

axes[1].bar(results.keys(), [v["RMSE"]/1e6 for v in results.values()], color=colors, edgecolor="white")
axes[1].set_title("RMSE (₹ Millions)"); axes[1].tick_params(axis="x", rotation=20)

axes[2].bar(results.keys(), [v["R2"] for v in results.values()], color=colors, edgecolor="white")
axes[2].set_title("R² Score"); axes[2].tick_params(axis="x", rotation=20)
axes[2].set_ylim(0, 1.1)
save_fig("13_model_metrics")

# ── Save metrics JSON ─────────────────────────────────────
metrics_out = {k: {"MAE": round(v["MAE"]), "RMSE": round(v["RMSE"]), "R2": round(v["R2"], 4)}
               for k, v in results.items()}
metrics_out["best_model"] = best_name
with open(os.path.join(MODELS, "model_metrics.json"), "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"\n✅ Forecasting complete. Outputs saved to /data/ and /visuals/")
print("=" * 60)
