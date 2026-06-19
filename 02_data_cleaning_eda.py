"""
============================================================
  Sales Data Analysis Dashboard 2025
  Script 02: Data Cleaning & Exploratory Data Analysis
  Author: Data Analytics Team
  Description: Full pipeline — cleaning, EDA, KPI computation,
               and chart generation saved to /visuals/
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────
BASE    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(BASE, "data")
VISUALS = os.path.join(BASE, "visuals")
os.makedirs(VISUALS, exist_ok=True)

# ── Colour palette ────────────────────────────────────────
PALETTE   = ["#1A3C6E", "#2E86AB", "#F24236", "#F4A261", "#2EC4B6",
             "#E9C46A", "#A8DADC", "#457B9D", "#E63946", "#06D6A0"]
DARK_BLUE = "#1A3C6E"
ACCENT    = "#2E86AB"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.edgecolor":    "#DEE2E6",
    "axes.grid":         True,
    "grid.color":        "#E9ECEF",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
})

def save_fig(name, tight=True):
    path = os.path.join(VISUALS, f"{name}.png")
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   📊 Saved: visuals/{name}.png")

# ══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 1 — Loading Raw Data")
print("=" * 60)

df_raw = pd.read_csv(os.path.join(DATA, "raw_sales_data.csv"))
print(f"   Raw shape      : {df_raw.shape}")
print(f"   Columns        : {list(df_raw.columns)}")
print(f"\n   Missing values :\n{df_raw.isnull().sum()[df_raw.isnull().sum() > 0]}")
print(f"\n   Duplicates     : {df_raw.duplicated().sum()}")

# ══════════════════════════════════════════════════════════
# 2. DATA CLEANING
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2 — Data Cleaning")
print("=" * 60)

df = df_raw.copy()

# 2a. Parse dates
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# 2b. Remove duplicates (keep first occurrence)
before = len(df)
df.drop_duplicates(subset="Order_ID", keep="first", inplace=True)
print(f"   Duplicates removed : {before - len(df)}")

# 2c. Impute missing values
df["Discount"].fillna(0.0, inplace=True)                                # assume no discount
df["Salesperson"].fillna("Unassigned", inplace=True)
df["City"].fillna(df["State"].apply(lambda s: s.split()[0] + " City"), inplace=True)

# 2d. Recalculate Revenue, Profit, Profit_Margin to ensure consistency
df["Revenue"]       = (df["Unit_Price"] * df["Quantity"] * (1 - df["Discount"])).round(2)
df["Profit"]        = (df["Revenue"] - df["Cost"]).round(2)
df["Profit_Margin"] = ((df["Profit"] / df["Revenue"]) * 100).round(2)

# 2e. Outlier detection — Revenue (IQR method)
Q1, Q3 = df["Revenue"].quantile(0.25), df["Revenue"].quantile(0.75)
IQR     = Q3 - Q1
lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
outliers_mask = (df["Revenue"] < lower) | (df["Revenue"] > upper)
print(f"   Revenue outliers   : {outliers_mask.sum()} (capped, not removed)")
df["Revenue"] = df["Revenue"].clip(lower=max(0, lower), upper=upper)

# 2f. Feature Engineering
df["Year"]          = df["Order_Date"].dt.year
df["Month"]         = df["Order_Date"].dt.month
df["Month_Name"]    = df["Order_Date"].dt.strftime("%b")
df["Quarter"]       = df["Order_Date"].dt.quarter.map({1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
df["Year_Quarter"]  = df["Year"].astype(str) + "-" + df["Quarter"]
df["Year_Month"]    = df["Order_Date"].dt.to_period("M").astype(str)
df["Week"]          = df["Order_Date"].dt.isocalendar().week.astype(int)
df["Day_of_Week"]   = df["Order_Date"].dt.day_name()
df["Is_Weekend"]    = df["Day_of_Week"].isin(["Saturday", "Sunday"])
df["Revenue_Band"]  = pd.cut(df["Revenue"],
                              bins=[0, 5000, 20000, 60000, float("inf")],
                              labels=["Low", "Medium", "High", "Premium"])

print(f"\n   Cleaned shape      : {df.shape}")
print(f"   Remaining nulls    : {df.isnull().sum().sum()}")

# Save cleaned data
df.to_csv(os.path.join(DATA, "cleaned_sales_data.csv"), index=False)
print(f"\n   ✅ Cleaned data saved → data/cleaned_sales_data.csv")

# ══════════════════════════════════════════════════════════
# 3. KPI COMPUTATION
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 3 — KPI Metrics")
print("=" * 60)

total_revenue  = df["Revenue"].sum()
total_profit   = df["Profit"].sum()
total_orders   = df["Order_ID"].nunique()
avg_order_val  = df["Revenue"].mean()
profit_margin  = (total_profit / total_revenue) * 100
top_product    = df.groupby("Product_Name")["Revenue"].sum().idxmax()
top_region     = df.groupby("Region")["Revenue"].sum().idxmax()
top_segment    = df.groupby("Customer_Segment")["Revenue"].sum().idxmax()

rev_2024 = df[df["Year"] == 2024]["Revenue"].sum()
rev_2025 = df[df["Year"] == 2025]["Revenue"].sum()
yoy_growth = ((rev_2025 - rev_2024) / rev_2024) * 100

kpis = {
    "Total Revenue (₹)":        f"₹{total_revenue:,.0f}",
    "Total Profit (₹)":         f"₹{total_profit:,.0f}",
    "Profit Margin (%)":        f"{profit_margin:.1f}%",
    "Total Orders":             f"{total_orders:,}",
    "Avg Order Value (₹)":      f"₹{avg_order_val:,.0f}",
    "Top Product":              top_product,
    "Top Region":               top_region,
    "Top Customer Segment":     top_segment,
    "YoY Revenue Growth (%)":   f"{yoy_growth:.1f}%",
    "2024 Revenue (₹)":         f"₹{rev_2024:,.0f}",
    "2025 Revenue (₹)":         f"₹{rev_2025:,.0f}",
}

for k, v in kpis.items():
    print(f"   {k:<30} {v}")

# ══════════════════════════════════════════════════════════
# 4. EDA VISUALISATIONS
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 4 — Generating EDA Visualisations")
print("=" * 60)

MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── 4.1 KPI Summary Card ──────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 6))
fig.suptitle("Sales Dashboard 2025 — KPI Summary", fontsize=16,
             fontweight="bold", color=DARK_BLUE, y=1.02)
kpi_items = [
    ("Total Revenue",   f"₹{total_revenue/1e7:.1f} Cr", "#1A3C6E", "#E8F4FD"),
    ("Total Profit",    f"₹{total_profit/1e7:.1f} Cr",  "#2E86AB", "#E8F4FD"),
    ("Profit Margin",   f"{profit_margin:.1f}%",         "#2EC4B6", "#E8F8F5"),
    ("Total Orders",    f"{total_orders:,}",             "#F4A261", "#FFF5EC"),
    ("Avg Order Value", f"₹{avg_order_val/1000:.1f}K",  "#E63946", "#FFF0F0"),
    ("YoY Growth",      f"{yoy_growth:.1f}%",            "#06D6A0", "#E8FFF5"),
    ("Top Region",      top_region,                      "#457B9D", "#EEF3F8"),
    ("Top Segment",     top_segment,                     "#E9C46A", "#FFFBEC"),
    ("2024 Revenue",    f"₹{rev_2024/1e7:.1f} Cr",      "#6C757D", "#F8F9FA"),
    ("2025 Revenue",    f"₹{rev_2025/1e7:.1f} Cr",      "#198754", "#EAFAF1"),
]
for ax, (label, val, text_color, bg) in zip(axes.flat, kpi_items):
    ax.set_facecolor(bg)
    ax.text(0.5, 0.62, val, ha="center", va="center", fontsize=17,
            fontweight="bold", color=text_color, transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha="center", va="center", fontsize=9.5,
            color="#6C757D", transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color); spine.set_linewidth(1.5)
    ax.set_xticks([]); ax.set_yticks([])
save_fig("01_kpi_summary")

# ── 4.2 Monthly Revenue Trend ─────────────────────────────
monthly = (df.groupby(["Year", "Month_Name"])
             .agg(Revenue=("Revenue", "sum"), Profit=("Profit", "sum"))
             .reset_index())
monthly["Month_Name"] = pd.Categorical(monthly["Month_Name"], categories=MONTH_ORDER, ordered=True)
monthly = monthly.sort_values(["Year", "Month_Name"])

fig, ax = plt.subplots(figsize=(14, 5))
for yr, color in zip([2024, 2025], [DARK_BLUE, ACCENT]):
    sub = monthly[monthly["Year"] == yr]
    ax.plot(sub["Month_Name"], sub["Revenue"] / 1e6,
            marker="o", linewidth=2.5, markersize=7, color=color, label=str(yr))
    ax.fill_between(range(len(sub)), sub["Revenue"].values / 1e6,
                    alpha=0.08, color=color)

ax.set_title("Monthly Revenue Trend — 2024 vs 2025")
ax.set_xlabel("Month"); ax.set_ylabel("Revenue (₹ Millions)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}M"))
ax.legend(); ax.set_xticks(range(12)); ax.set_xticklabels(MONTH_ORDER)
save_fig("02_monthly_revenue_trend")

# ── 4.3 Revenue by Region ─────────────────────────────────
region_rev = (df.groupby("Region")
                .agg(Revenue=("Revenue","sum"), Profit=("Profit","sum"), Orders=("Order_ID","nunique"))
                .reset_index().sort_values("Revenue", ascending=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Regional Performance Overview", fontsize=14, fontweight="bold", color=DARK_BLUE)

bars = axes[0].bar(region_rev["Region"], region_rev["Revenue"]/1e6,
                   color=PALETTE[:4], edgecolor="white", linewidth=0.8)
for b in bars:
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+2,
                 f"₹{b.get_height():.0f}M", ha="center", fontsize=8.5, fontweight="bold")
axes[0].set_title("Revenue by Region"); axes[0].set_ylabel("Revenue (₹ Millions)")

axes[1].bar(region_rev["Region"], region_rev["Profit"]/1e6,
            color=PALETTE[4:8], edgecolor="white")
axes[1].set_title("Profit by Region"); axes[1].set_ylabel("Profit (₹ Millions)")

axes[2].pie(region_rev["Orders"], labels=region_rev["Region"],
            autopct="%1.1f%%", colors=PALETTE[:4], startangle=90,
            wedgeprops={"edgecolor":"white","linewidth":1.5})
axes[2].set_title("Orders Distribution")
save_fig("03_regional_performance")

# ── 4.4 Product Category Performance ─────────────────────
cat_perf = (df.groupby("Product_Category")
              .agg(Revenue=("Revenue","sum"), Profit=("Profit","sum"),
                   Orders=("Order_ID","nunique"), Avg_Margin=("Profit_Margin","mean"))
              .reset_index().sort_values("Revenue", ascending=False))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Product Category Analysis", fontsize=14, fontweight="bold", color=DARK_BLUE)

axes[0,0].barh(cat_perf["Product_Category"], cat_perf["Revenue"]/1e6,
               color=PALETTE[:5], edgecolor="white")
axes[0,0].set_title("Revenue by Category"); axes[0,0].set_xlabel("₹ Millions")

axes[0,1].barh(cat_perf["Product_Category"], cat_perf["Profit"]/1e6,
               color=PALETTE[2:7], edgecolor="white")
axes[0,1].set_title("Profit by Category"); axes[0,1].set_xlabel("₹ Millions")

axes[1,0].bar(cat_perf["Product_Category"], cat_perf["Avg_Margin"],
              color=PALETTE[1:6], edgecolor="white")
axes[1,0].set_title("Average Profit Margin %"); axes[1,0].set_ylabel("%")
axes[1,0].tick_params(axis="x", rotation=20)

axes[1,1].scatter(cat_perf["Revenue"]/1e6, cat_perf["Avg_Margin"],
                  s=cat_perf["Orders"]/20, c=PALETTE[:5], alpha=0.85, edgecolors="white")
for _, row in cat_perf.iterrows():
    axes[1,1].annotate(row["Product_Category"], (row["Revenue"]/1e6, row["Avg_Margin"]),
                       fontsize=8, xytext=(5,3), textcoords="offset points")
axes[1,1].set_title("Revenue vs Margin (bubble = orders)")
axes[1,1].set_xlabel("Revenue (₹M)"); axes[1,1].set_ylabel("Avg Margin %")
save_fig("04_category_performance")

# ── 4.5 Top 10 Products ───────────────────────────────────
top10 = (df.groupby("Product_Name")["Revenue"].sum()
           .sort_values(ascending=False).head(10).reset_index())
bottom10 = (df.groupby("Product_Name")["Revenue"].sum()
              .sort_values(ascending=True).head(10).reset_index())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Product Performance — Top 10 & Bottom 10", fontsize=14,
             fontweight="bold", color=DARK_BLUE)
axes[0].barh(top10["Product_Name"][::-1], top10["Revenue"][::-1]/1e6,
             color=ACCENT, edgecolor="white")
axes[0].set_title("Top 10 Products by Revenue"); axes[0].set_xlabel("Revenue (₹M)")

axes[1].barh(bottom10["Product_Name"], bottom10["Revenue"]/1e6,
             color="#E63946", edgecolor="white")
axes[1].set_title("Bottom 10 Products by Revenue"); axes[1].set_xlabel("Revenue (₹M)")
save_fig("05_product_top_bottom10")

# ── 4.6 Customer Segmentation ─────────────────────────────
seg = (df.groupby("Customer_Segment")
         .agg(Revenue=("Revenue","sum"), Orders=("Order_ID","nunique"),
              Avg_Val=("Revenue","mean"), Profit=("Profit","sum"))
         .reset_index())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Customer Segmentation Analysis", fontsize=14, fontweight="bold", color=DARK_BLUE)

axes[0].pie(seg["Revenue"], labels=seg["Customer_Segment"], autopct="%1.1f%%",
            colors=PALETTE[:5], startangle=90, wedgeprops={"edgecolor":"white","linewidth":1.5})
axes[0].set_title("Revenue Share by Segment")

axes[1].bar(seg["Customer_Segment"], seg["Avg_Val"]/1000,
            color=PALETTE[1:6], edgecolor="white")
axes[1].set_title("Average Order Value by Segment")
axes[1].set_ylabel("₹ Thousands"); axes[1].tick_params(axis="x", rotation=15)

axes[2].bar(seg["Customer_Segment"], seg["Orders"],
            color=PALETTE[3:8], edgecolor="white")
axes[2].set_title("Number of Orders by Segment")
axes[2].set_ylabel("Orders"); axes[2].tick_params(axis="x", rotation=15)
save_fig("06_customer_segmentation")

# ── 4.7 Payment & Shipping Analysis ──────────────────────
pay = df["Payment_Method"].value_counts().reset_index()
ship = df["Shipping_Mode"].value_counts().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Payment & Shipping Mode Analysis", fontsize=14, fontweight="bold", color=DARK_BLUE)

axes[0].bar(pay["Payment_Method"], pay["count"], color=PALETTE[:6], edgecolor="white")
axes[0].set_title("Orders by Payment Method")
axes[0].set_ylabel("Number of Orders"); axes[0].tick_params(axis="x", rotation=20)

axes[1].pie(ship["count"], labels=ship["Shipping_Mode"], autopct="%1.1f%%",
            colors=PALETTE[2:6], startangle=90, wedgeprops={"edgecolor":"white","linewidth":1.5})
axes[1].set_title("Shipping Mode Distribution")
save_fig("07_payment_shipping")

# ── 4.8 Quarterly Revenue ─────────────────────────────────
qtr = (df.groupby(["Year","Quarter"])
         .agg(Revenue=("Revenue","sum"), Profit=("Profit","sum"))
         .reset_index())
qtr["Label"] = qtr["Year"].astype(str) + " " + qtr["Quarter"]

fig, ax = plt.subplots(figsize=(12, 5))
x = range(len(qtr))
bars = ax.bar(x, qtr["Revenue"]/1e6, color=[DARK_BLUE if y==2024 else ACCENT for y in qtr["Year"]],
              edgecolor="white", linewidth=0.8)
ax.plot(x, qtr["Profit"]/1e6, marker="D", color="#E63946",
        linewidth=2, markersize=6, label="Profit", zorder=5)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
            f"₹{b.get_height():.0f}M", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(qtr["Label"], rotation=30)
ax.set_title("Quarterly Revenue & Profit — 2024 vs 2025")
ax.set_ylabel("₹ Millions")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=DARK_BLUE, label="2024 Revenue"),
                   Patch(color=ACCENT, label="2025 Revenue"),
                   plt.Line2D([0],[0], color="#E63946", marker="D", label="Profit")])
save_fig("08_quarterly_revenue")

# ── 4.9 Discount Impact on Profit Margin ─────────────────
disc_bins = pd.cut(df["Discount"], bins=[-0.01, 0, 0.05, 0.10, 0.15, 0.25],
                   labels=["No Discount","5%","10%","15%","20-25%"])
disc_impact = df.groupby(disc_bins, observed=True)["Profit_Margin"].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Discount Impact Analysis", fontsize=14, fontweight="bold", color=DARK_BLUE)
axes[0].bar(disc_impact["Discount"], disc_impact["Profit_Margin"],
            color=PALETTE[:5], edgecolor="white")
axes[0].set_title("Avg Profit Margin by Discount Level")
axes[0].set_xlabel("Discount Level"); axes[0].set_ylabel("Avg Profit Margin %")

axes[1].scatter(df.sample(3000, random_state=42)["Discount"],
                df.sample(3000, random_state=42)["Profit_Margin"],
                alpha=0.3, color=ACCENT, s=15)
axes[1].set_title("Discount vs Profit Margin (scatter)")
axes[1].set_xlabel("Discount Rate"); axes[1].set_ylabel("Profit Margin %")
save_fig("09_discount_impact")

# ── 4.10 Correlation Heatmap ──────────────────────────────
num_cols = ["Quantity","Unit_Price","Discount","Revenue","Cost","Profit","Profit_Margin"]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5, square=True,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap")
save_fig("10_correlation_heatmap")

# ══════════════════════════════════════════════════════════
print("\n✅ EDA complete. All visuals saved to /visuals/")
print("=" * 60)
