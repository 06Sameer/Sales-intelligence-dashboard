"""
============================================================
  Sales Data Analysis Dashboard 2025
  Script 01: Dataset Generation
  Author: Data Analytics Team
  Description: Generates 50,000+ realistic sales records
               spanning Jan 2024 - Dec 2025 with seasonal
               variations, regional patterns, and promotions.
============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ── Reproducibility ──────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── Constants ────────────────────────────────────────────
N_RECORDS = 52000

REGIONS = {
    "North": ["Delhi", "Punjab", "Haryana", "Uttar Pradesh", "Himachal Pradesh"],
    "South": ["Karnataka", "Tamil Nadu", "Andhra Pradesh", "Kerala", "Telangana"],
    "East":  ["West Bengal", "Odisha", "Bihar", "Jharkhand", "Assam"],
    "West":  ["Maharashtra", "Gujarat", "Rajasthan", "Goa", "Madhya Pradesh"],
}

STATE_CITIES = {
    "Delhi":            ["New Delhi", "Dwarka", "Rohini"],
    "Punjab":           ["Chandigarh", "Ludhiana", "Amritsar"],
    "Haryana":          ["Gurugram", "Faridabad", "Panipat"],
    "Uttar Pradesh":    ["Lucknow", "Agra", "Kanpur"],
    "Himachal Pradesh": ["Shimla", "Manali", "Dharamshala"],
    "Karnataka":        ["Bengaluru", "Mysuru", "Hubli"],
    "Tamil Nadu":       ["Chennai", "Coimbatore", "Madurai"],
    "Andhra Pradesh":   ["Visakhapatnam", "Vijayawada", "Guntur"],
    "Kerala":           ["Kochi", "Thiruvananthapuram", "Kozhikode"],
    "Telangana":        ["Hyderabad", "Warangal", "Karimnagar"],
    "West Bengal":      ["Kolkata", "Howrah", "Durgapur"],
    "Odisha":           ["Bhubaneswar", "Cuttack", "Rourkela"],
    "Bihar":            ["Patna", "Gaya", "Muzaffarpur"],
    "Jharkhand":        ["Ranchi", "Jamshedpur", "Dhanbad"],
    "Assam":            ["Guwahati", "Dibrugarh", "Silchar"],
    "Maharashtra":      ["Mumbai", "Pune", "Nagpur"],
    "Gujarat":          ["Ahmedabad", "Surat", "Vadodara"],
    "Rajasthan":        ["Jaipur", "Jodhpur", "Udaipur"],
    "Goa":              ["Panaji", "Margao", "Vasco da Gama"],
    "Madhya Pradesh":   ["Bhopal", "Indore", "Gwalior"],
}

PRODUCTS = {
    "Electronics": {
        "Laptops":       [("ProBook X1", 55000), ("UltraSlim Z3", 72000), ("WorkStation Pro", 95000)],
        "Smartphones":   [("Galaxy S24", 45000), ("iPhone 15", 79000), ("Pixel 8", 52000)],
        "Accessories":   [("Wireless Earbuds", 3500), ("Smart Watch", 12000), ("USB Hub", 1800)],
    },
    "Furniture": {
        "Office Chairs": [("ErgoComfort Pro", 18000), ("Executive Chair", 25000), ("Mesh Back Chair", 9500)],
        "Desks":         [("Standing Desk", 32000), ("Corner Workstation", 27000), ("Compact Desk", 12000)],
        "Storage":       [("Filing Cabinet", 8500), ("Bookshelf Premium", 6200), ("Storage Unit", 4800)],
    },
    "Office Supplies": {
        "Stationery":    [("Premium Pen Set", 850), ("Notebook Bundle", 450), ("Marker Kit", 320)],
        "Paper":         [("A4 Ream 500", 280), ("Photo Paper Pack", 650), ("Carbon Paper", 180)],
        "Organizers":    [("Desk Organizer", 1200), ("File Folder Set", 380), ("Label Maker", 2100)],
    },
    "Clothing": {
        "Formal Wear":   [("Business Suit", 8500), ("Formal Shirt Pack", 2200), ("Trousers Set", 3400)],
        "Casual Wear":   [("Cotton T-Shirts", 1200), ("Denim Jeans", 2800), ("Casual Blazer", 4500)],
        "Accessories":   [("Leather Belt", 1500), ("Tie Collection", 1800), ("Formal Shoes", 5500)],
    },
    "Sports & Fitness": {
        "Equipment":     [("Treadmill Pro", 45000), ("Dumbbell Set", 8500), ("Yoga Mat Premium", 1200)],
        "Apparel":       [("Sports Jersey", 1800), ("Running Shoes", 4500), ("Compression Shorts", 1200)],
        "Nutrition":     [("Whey Protein 2kg", 3200), ("BCAA Supplement", 1800), ("Multivitamin Pack", 950)],
    },
}

SALESPERSONS = [
    "Rahul Sharma", "Priya Patel", "Amit Kumar", "Sneha Singh",
    "Vikram Nair", "Ananya Iyer", "Rohan Gupta", "Kavya Reddy",
    "Arjun Mehta", "Deepika Joshi", "Sanjay Verma", "Neha Agarwal",
    "Kiran Rao", "Pooja Mishra", "Suresh Pillai", "Riya Bose",
]

CUSTOMER_SEGMENTS = ["Corporate", "SME", "Retail", "Government", "Startup"]

PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery", "EMI"]

SHIPPING_MODES = ["Standard", "Express", "Same Day", "Economy"]

# ── Seasonal multipliers (month 1–12) ────────────────────
SEASONAL_WEIGHTS = [0.75, 0.70, 0.85, 0.90, 0.88, 0.80,
                    0.82, 0.85, 0.95, 1.10, 1.35, 1.50]  # Nov-Dec peak

# ── Regional performance weights ─────────────────────────
REGION_WEIGHTS = {"North": 0.32, "South": 0.30, "West": 0.28, "East": 0.10}


def generate_date(year_weights=None):
    """Generate a random date between Jan 2024 and Dec 2025."""
    start = datetime(2024, 1, 1)
    end   = datetime(2025, 12, 31)
    delta = end - start
    # Apply year bias: 2025 slightly more records (growth)
    rand_days = random.randint(0, delta.days)
    date = start + timedelta(days=rand_days)
    # Apply seasonal weight
    month_wt = SEASONAL_WEIGHTS[date.month - 1]
    if random.random() > month_wt * 0.6:
        return date
    return date


def generate_customer_id(idx):
    return f"CUST-{str(idx % 8000 + 1001).zfill(5)}"


def generate_customer_name(cust_id):
    first_names = ["Aarav", "Aditi", "Aditya", "Ananya", "Arjun", "Deepak",
                   "Divya", "Ishaan", "Kavya", "Kiran", "Meera", "Mohit",
                   "Neha", "Priya", "Rahul", "Rajesh", "Riya", "Rohan",
                   "Sneha", "Suresh", "Tanvi", "Vikram", "Vivek", "Zara"]
    last_names  = ["Agarwal", "Bose", "Chopra", "Gupta", "Iyer", "Joshi",
                   "Kumar", "Mehta", "Mishra", "Nair", "Patel", "Pillai",
                   "Rao", "Reddy", "Sharma", "Singh", "Verma"]
    seed_val = int(cust_id.split("-")[1])
    fn = first_names[seed_val % len(first_names)]
    ln = last_names[(seed_val * 7) % len(last_names)]
    return f"{fn} {ln}"


def build_records():
    records = []

    # Flatten product list
    all_products = []
    for category, sub_dict in PRODUCTS.items():
        for sub_cat, items in sub_dict.items():
            for prod_name, base_price in items:
                all_products.append({
                    "category": category,
                    "sub_category": sub_cat,
                    "product_name": prod_name,
                    "base_price": base_price,
                })

    region_list = list(REGION_WEIGHTS.keys())
    region_probs = list(REGION_WEIGHTS.values())

    for i in range(N_RECORDS):
        # ── Order basics ─────────────────────────────────
        order_id = f"ORD-{str(i + 10001).zfill(6)}"
        order_date = generate_date()
        month = order_date.month
        year  = order_date.year

        # ── Customer ─────────────────────────────────────
        cust_id   = generate_customer_id(i)
        cust_name = generate_customer_name(cust_id)
        cust_seg  = np.random.choice(
            CUSTOMER_SEGMENTS, p=[0.30, 0.25, 0.25, 0.10, 0.10]
        )

        # ── Geography ────────────────────────────────────
        region = np.random.choice(region_list, p=region_probs)
        state  = random.choice(REGIONS[region])
        city   = random.choice(STATE_CITIES[state])

        # ── Product ───────────────────────────────────────
        prod = random.choice(all_products)
        prod_id = f"PROD-{str(all_products.index(prod) + 101).zfill(4)}"

        # Price with seasonal & segment variation
        seasonal_factor = 1 + (SEASONAL_WEIGHTS[month - 1] - 0.9) * 0.05
        segment_factor  = {"Corporate": 1.15, "SME": 1.05, "Retail": 1.0,
                           "Government": 0.95, "Startup": 0.98}[cust_seg]
        unit_price = round(prod["base_price"] * seasonal_factor * segment_factor *
                           np.random.uniform(0.92, 1.08), 2)

        # Quantity
        qty_map = {"Electronics": (1, 3), "Furniture": (1, 2),
                   "Office Supplies": (2, 15), "Clothing": (1, 5),
                   "Sports & Fitness": (1, 4)}
        qty = random.randint(*qty_map[prod["category"]])

        # Discount (promotional months: Nov, Dec, Jan, festival)
        festival_months = {1, 10, 11, 12}
        if month in festival_months:
            discount = round(random.choice([0.05, 0.10, 0.15, 0.20, 0.25]), 2)
        else:
            discount = round(random.choice([0.0, 0.0, 0.05, 0.10]), 2)

        # Financials
        gross_revenue = round(unit_price * qty, 2)
        revenue       = round(gross_revenue * (1 - discount), 2)
        cost_pct      = {"Electronics": 0.68, "Furniture": 0.55,
                         "Office Supplies": 0.45, "Clothing": 0.50,
                         "Sports & Fitness": 0.60}[prod["category"]]
        cost          = round(revenue * cost_pct * np.random.uniform(0.95, 1.05), 2)
        profit        = round(revenue - cost, 2)
        profit_margin = round((profit / revenue) * 100, 2) if revenue > 0 else 0

        # Salesperson (region-biased)
        salesperson = random.choice(SALESPERSONS)

        # Payment & Shipping
        payment  = np.random.choice(PAYMENT_METHODS, p=[0.25, 0.20, 0.30, 0.10, 0.10, 0.05])
        shipping = np.random.choice(SHIPPING_MODES,  p=[0.45, 0.30, 0.10, 0.15])

        records.append({
            "Order_ID":        order_id,
            "Order_Date":      order_date.strftime("%Y-%m-%d"),
            "Customer_ID":     cust_id,
            "Customer_Name":   cust_name,
            "Customer_Segment": cust_seg,
            "Product_ID":      prod_id,
            "Product_Name":    prod["product_name"],
            "Product_Category": prod["category"],
            "Sub_Category":    prod["sub_category"],
            "Region":          region,
            "State":           state,
            "City":            city,
            "Salesperson":     salesperson,
            "Quantity":        qty,
            "Unit_Price":      unit_price,
            "Discount":        discount,
            "Revenue":         revenue,
            "Cost":            cost,
            "Profit":          profit,
            "Profit_Margin":   profit_margin,
            "Payment_Method":  payment,
            "Shipping_Mode":   shipping,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("=" * 60)
    print("  Generating Sales Dataset — 52,000 Records")
    print("=" * 60)

    df = build_records()
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df = df.sort_values("Order_Date").reset_index(drop=True)

    # Inject ~1% missing values realistically
    for col in ["Discount", "Salesperson", "City"]:
        mask = np.random.choice([True, False], size=len(df), p=[0.008, 0.992])
        df.loc[mask, col] = np.nan

    # Inject ~0.3% duplicates
    dup_rows = df.sample(frac=0.003, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_sales_data.csv")
    df.to_csv(out_path, index=False)

    print(f"\n✅ Dataset saved: {out_path}")
    print(f"   Total records : {len(df):,}")
    print(f"   Date range    : {df['Order_Date'].min().date()} → {df['Order_Date'].max().date()}")
    print(f"   Unique orders : {df['Order_ID'].nunique():,}")
    print(f"   Total revenue : ₹{df['Revenue'].sum():,.0f}")
    print(f"   Avg order val : ₹{df['Revenue'].mean():,.0f}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Duplicate rows: {dup_rows.shape[0]}")
