"""DataFix Demo Data Generator: Synthesizes realistic relational datasets
(Customers and Orders) with carefully injected quality errors across all issue types,
and outputs ground_truth.json for benchmarking detection and repair accuracy.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd


FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White"
]

DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "company.com", "techcorp.io"]

COUNTRIES_CANONICAL = ["United States", "Canada", "United Kingdom", "Germany", "France", "Australia"]

CATEGORIES = ["Electronics", "Home & Garden", "Apparel", "Books", "Health & Beauty", "Sports"]


def generate_clean_datasets(n_customers: int = 5000, n_orders: int = 5000, seed: int = 42):
    """Generate clean baseline customers and orders DataFrames."""
    rng = np.random.default_rng(seed)

    # 1. Customers
    customer_ids = np.arange(1001, 1001 + n_customers, dtype=int)
    firsts = rng.choice(FIRST_NAMES, size=n_customers)
    lasts = rng.choice(LAST_NAMES, size=n_customers)
    names = [f"{f} {l}" for f, l in zip(firsts, lasts)]
    
    emails = [
        f"{f.lower()}.{l.lower()}{rng.integers(1, 999)}@{rng.choice(DOMAINS)}"
        for f, l in zip(firsts, lasts)
    ]
    
    phones = [
        f"+1-{rng.integers(200, 999)}-{rng.integers(100, 999)}-{rng.integers(1000, 9999)}"
        for _ in range(n_customers)
    ]
    
    ages = rng.integers(18, 75, size=n_customers)
    countries = rng.choice(COUNTRIES_CANONICAL, size=n_customers, p=[0.5, 0.15, 0.15, 0.08, 0.07, 0.05])
    segments = rng.choice(["Standard", "Silver", "Gold", "Platinum"], size=n_customers, p=[0.6, 0.25, 0.1, 0.05])
    
    # Registration dates between 2020-01-01 and 2023-12-31
    base_date = pd.Timestamp("2020-01-01")
    reg_offsets = rng.integers(0, 1400, size=n_customers)
    created_dates = [base_date + pd.Timedelta(days=int(off)) for off in reg_offsets]
    created_str = [d.strftime("%Y-%m-%d") for d in created_dates]

    df_customers = pd.DataFrame({
        "customer_id": customer_ids,
        "name": names,
        "email": emails,
        "phone": phones,
        "age": ages,
        "country": countries,
        "segment": segments,
        "signup_date": created_str
    })

    # 2. Orders
    order_ids = np.arange(50001, 50001 + n_orders, dtype=int)
    # Most orders link to existing customers
    order_customer_ids = rng.choice(customer_ids, size=n_orders)
    categories = rng.choice(CATEGORIES, size=n_orders)
    quantities = rng.integers(1, 10, size=n_orders)
    unit_prices = np.round(rng.uniform(10.0, 500.0, size=n_orders), 2)
    totals = np.round(quantities * unit_prices, 2)

    order_base_date = pd.Timestamp("2022-01-01")
    order_offsets = rng.integers(0, 700, size=n_orders)
    order_dates = [order_base_date + pd.Timedelta(days=int(off)) for off in order_offsets]
    delivery_delays = rng.integers(2, 14, size=n_orders)
    delivery_dates = [od + pd.Timedelta(days=int(d)) for od, d in zip(order_dates, delivery_delays)]

    order_dates_str = [d.strftime("%Y-%m-%d") for d in order_dates]
    delivery_dates_str = [d.strftime("%Y-%m-%d") for d in delivery_dates]

    df_orders = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": order_customer_ids,
        "category": categories,
        "quantity": quantities,
        "unit_price": unit_prices,
        "total_amount": totals,
        "order_date": order_dates_str,
        "delivery_date": delivery_dates_str,
        "status": rng.choice(["Delivered", "Shipped", "Processing"], size=n_orders, p=[0.7, 0.2, 0.1])
    })

    return df_customers, df_orders


def inject_errors(df_cust: pd.DataFrame, df_ord: pd.DataFrame, seed: int = 42):
    """Inject specific, tracked errors into customers and orders tables."""
    rng = np.random.default_rng(seed)
    injections: List[Dict[str, Any]] = []

    cust = df_cust.copy()
    ordr = df_ord.copy()

    # Assign _row_id to track row index precisely
    cust["_row_id"] = np.arange(len(cust))
    ordr["_row_id"] = np.arange(len(ordr))

    injection_counter = 0

    def add_inj(table, row_id, column, issue_type, clean_val, corrupted_val, details):
        nonlocal injection_counter
        injection_counter += 1
        injections.append({
            "injection_id": f"INJ_{injection_counter:04d}",
            "table": table,
            "row_id": int(row_id),
            "column": column,
            "issue_type": issue_type,
            "clean_value": str(clean_val),
            "corrupted_value": str(corrupted_val) if corrupted_val is not None else "None",
            "details": details
        })

    # Ensure columns that receive mixed types are object dtype
    cust["email"] = cust["email"].astype(object)
    cust["phone"] = cust["phone"].astype(object)
    cust["country"] = cust["country"].astype(object)
    cust["segment"] = cust["segment"].astype(object)
    cust["signup_date"] = cust["signup_date"].astype(object)
    cust["age"] = cust["age"].astype(object)

    ordr["total_amount"] = ordr["total_amount"].astype(object)
    ordr["unit_price"] = ordr["unit_price"].astype(object)
    ordr["delivery_date"] = ordr["delivery_date"].astype(object)
    ordr["customer_id"] = ordr["customer_id"].astype(object)

    n_c = len(cust)
    n_o = len(ordr)

    # Scale injection counts if dataset is small
    n_null_email = max(3, min(25, int(n_c * 0.10)))
    n_null_phone = max(3, min(30, int(n_c * 0.10)))
    n_null_deliv = max(3, min(20, int(n_o * 0.10)))
    n_exact_dup = max(2, min(15, int(n_c * 0.05)))
    n_fuzzy_dup = max(2, min(15, int(n_c * 0.05)))
    n_date_err = max(3, min(25, int(n_c * 0.08)))
    n_email_err = max(3, min(20, int(n_c * 0.08)))
    n_num_text = max(3, min(25, int(n_o * 0.08)))
    n_cat_usa = max(3, min(35, int(n_c * 0.10)))
    n_cat_seg = max(3, min(20, int(n_c * 0.08)))
    n_outlier_age = max(2, min(15, int(n_c * 0.05)))
    n_outlier_price = max(2, min(12, int(n_o * 0.05)))
    n_orphan_fk = max(2, min(25, int(n_o * 0.08)))
    n_cross_date = max(2, min(20, int(n_o * 0.08)))
    n_calc_err = max(2, min(25, int(n_o * 0.08)))

    # =========================================================================
    # 1. MISSING VALUES & DISGUISED NULLS
    # =========================================================================
    null_cust_rows = rng.choice(cust["_row_id"].values, size=n_null_email, replace=False)
    for r in null_cust_rows:
        orig = cust.at[r, "email"]
        cust.at[r, "email"] = np.nan
        add_inj("customers", r, "email", "missing_value", orig, np.nan, "Standard NaN null")

    disguised_tokens = ["N/A", "null", "-", "?", "None", "unknown", "missing"]
    available_phone_rows = [r for r in cust["_row_id"].values if r not in set(null_cust_rows)]
    phone_null_rows = rng.choice(available_phone_rows, size=min(n_null_phone, len(available_phone_rows)), replace=False)
    for i, r in enumerate(phone_null_rows):
        token = disguised_tokens[i % len(disguised_tokens)]
        orig = cust.at[r, "phone"]
        cust.at[r, "phone"] = token
        add_inj("customers", r, "phone", "missing_value", orig, token, f"Disguised null '{token}'")

    ord_null_rows = rng.choice(ordr["_row_id"].values, size=n_null_deliv, replace=False)
    for r in ord_null_rows:
        orig = ordr.at[r, "delivery_date"]
        val = "N/A" if r % 2 == 0 else np.nan
        ordr.at[r, "delivery_date"] = val
        add_inj("orders", r, "delivery_date", "missing_value", orig, val, "Missing delivery date")

    # =========================================================================
    # 2. DUPLICATES (EXACT AND FUZZY)
    # =========================================================================
    dup_pool = cust["_row_id"].values[:max(10, int(n_c * 0.5))]
    exact_dup_source = rng.choice(dup_pool, size=min(n_exact_dup, len(dup_pool)), replace=False)
    for src_r in exact_dup_source:
        dup_row = cust.loc[cust["_row_id"] == src_r].copy()
        new_r = len(cust)
        dup_row["_row_id"] = new_r
        cust = pd.concat([cust, dup_row], ignore_index=True)
        add_inj("customers", new_r, "ALL", "duplicate_row", f"Row {src_r}", f"Row {new_r}", "Exact duplicate row")

    fuzzy_pool = [r for r in cust["_row_id"].values[:n_c] if r not in set(exact_dup_source)]
    fuzzy_sources = rng.choice(fuzzy_pool, size=min(n_fuzzy_dup, len(fuzzy_pool)), replace=False)
    for src_r in fuzzy_sources:
        fuzzy_row = cust.loc[cust["_row_id"] == src_r].copy()
        orig_name = str(fuzzy_row["name"].values[0])
        corrupted_name = orig_name[:-1] + orig_name[-1] * 2 if len(orig_name) > 3 else orig_name + "z"
        new_r = len(cust)
        fuzzy_row["_row_id"] = new_r
        fuzzy_row["name"] = corrupted_name
        cust = pd.concat([cust, fuzzy_row], ignore_index=True)
        add_inj("customers", new_r, "name", "duplicate_row", orig_name, corrupted_name, "Fuzzy duplicate name variant")

    # =========================================================================
    # 3. INVALID FORMATS
    # =========================================================================
    date_err_rows = rng.choice(cust["_row_id"].values[:n_c], size=min(n_date_err, n_c), replace=False)
    for i, r in enumerate(date_err_rows):
        orig = cust.at[r, "signup_date"]
        mode = i % 4
        if mode == 0:
            dt = pd.to_datetime(orig)
            corrupted = dt.strftime("%d/%m/%Y")
        elif mode == 1:
            dt = pd.to_datetime(orig)
            corrupted = dt.strftime("%b %d, %Y")
        elif mode == 2:
            corrupted = str(orig).replace("-", ".")
        else:
            corrupted = "2022-02-31"
        cust.at[r, "signup_date"] = corrupted
        add_inj("customers", r, "signup_date", "invalid_format", orig, corrupted, "Inconsistent date format")

    valid_email_rows = [r for r in cust["_row_id"].values[:n_c] if r not in set(null_cust_rows) and pd.notna(cust.at[r, "email"])]
    email_err_rows = rng.choice(valid_email_rows, size=min(n_email_err, len(valid_email_rows)), replace=False)
    for i, r in enumerate(email_err_rows):
        orig = str(cust.at[r, "email"])
        if i % 3 == 0:
            corrupted = orig.replace("@", "")
        elif i % 3 == 1:
            corrupted = orig + "..com"
        else:
            corrupted = orig.replace(".", " ")
        cust.at[r, "email"] = corrupted
        add_inj("customers", r, "email", "invalid_format", orig, corrupted, "Malformed email address")

    num_text_rows = rng.choice(ordr["_row_id"].values[:n_o], size=min(n_num_text, n_o), replace=False)
    for i, r in enumerate(num_text_rows):
        orig = ordr.at[r, "total_amount"]
        if i % 2 == 0:
            corrupted = f"${float(orig):,.2f}"
        else:
            corrupted = f"USD {orig}"
        ordr.at[r, "total_amount"] = corrupted
        add_inj("orders", r, "total_amount", "invalid_format", orig, corrupted, "Numeric with currency symbol/string")

    # =========================================================================
    # 4. INCONSISTENT CATEGORIES
    # =========================================================================
    usa_rows = cust[cust["country"] == "United States"]["_row_id"].values
    if len(usa_rows) > 0:
        chosen_usa = rng.choice(usa_rows, size=min(n_cat_usa, len(usa_rows)), replace=False)
        usa_variants = ["USA", "U.S.A.", "us", "  United States  ", "usa"]
        for i, r in enumerate(chosen_usa):
            var = usa_variants[i % len(usa_variants)]
            orig = cust.at[r, "country"]
            cust.at[r, "country"] = var
            add_inj("customers", r, "country", "inconsistent_category", "United States", var, "Country category variant")

    seg_rows = rng.choice(cust["_row_id"].values[:n_c], size=min(n_cat_seg, n_c), replace=False)
    for i, r in enumerate(seg_rows):
        orig = cust.at[r, "segment"]
        corrupted = orig.lower() if i % 2 == 0 else f"  {orig.upper()}  "
        cust.at[r, "segment"] = corrupted
        add_inj("customers", r, "segment", "inconsistent_category", orig, corrupted, "Segment casing/whitespace variant")

    # =========================================================================
    # 5. OUTLIERS (UNIVARIATE & MULTIVARIATE)
    # =========================================================================
    age_outlier_rows = rng.choice(cust["_row_id"].values[:n_c], size=min(n_outlier_age, n_c), replace=False)
    for i, r in enumerate(age_outlier_rows):
        orig = cust.at[r, "age"]
        corrupted = int(rng.integers(180, 260)) if i % 2 == 0 else -abs(int(orig))
        cust.at[r, "age"] = corrupted
        add_inj("customers", r, "age", "outlier", orig, corrupted, "Impossible age value outlier")

    price_outlier_rows = rng.choice(ordr["_row_id"].values[:n_o], size=min(n_outlier_price, n_o), replace=False)
    for i, r in enumerate(price_outlier_rows):
        orig = ordr.at[r, "unit_price"]
        corrupted = 49999.0 if i % 2 == 0 else -250.0
        ordr.at[r, "unit_price"] = corrupted
        add_inj("orders", r, "unit_price", "outlier", orig, corrupted, "Extreme order unit price outlier")

    # =========================================================================
    # 6. RELATIONSHIP VIOLATIONS
    # =========================================================================
    orphan_rows = rng.choice(ordr["_row_id"].values[:n_o], size=min(n_orphan_fk, n_o), replace=False)
    for i, r in enumerate(orphan_rows):
        orig = ordr.at[r, "customer_id"]
        corrupted = 99000 + i
        ordr.at[r, "customer_id"] = corrupted
        add_inj("orders", r, "customer_id", "relationship_violation", orig, corrupted, "Orphan foreign key: customer does not exist")

    avail_cross_date = [r for r in ordr["_row_id"].values[:n_o] if r not in set(ord_null_rows)]
    cross_date_rows = rng.choice(avail_cross_date, size=min(n_cross_date, len(avail_cross_date)), replace=False)
    for r in cross_date_rows:
        orig_ord_date = ordr.at[r, "order_date"]
        orig_deliv_date = ordr.at[r, "delivery_date"]
        try:
            bad_deliv = (pd.to_datetime(orig_ord_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            ordr.at[r, "delivery_date"] = bad_deliv
            add_inj("orders", r, "delivery_date", "relationship_violation", orig_deliv_date, bad_deliv, "Delivery date precedes order date")
        except Exception:
            pass

    avail_calc = [r for r in ordr["_row_id"].values[:n_o] if r not in set(num_text_rows)]
    calc_rows = rng.choice(avail_calc, size=min(n_calc_err, len(avail_calc)), replace=False)
    for r in calc_rows:
        orig_total = ordr.at[r, "total_amount"]
        corrupted_total = float(orig_total) + 150.0
        ordr.at[r, "total_amount"] = corrupted_total
        add_inj("orders", r, "total_amount", "relationship_violation", orig_total, corrupted_total, "Total amount mismatch with quantity * unit_price")

    return cust, ordr, injections


def generate_demo_files(output_dir: str = "data", seed: int = 42, n_rows: int = 5000):
    """Generate demo CSVs and ground_truth.json."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_clean_cust, df_clean_ord = generate_clean_datasets(n_customers=n_rows, n_orders=n_rows, seed=seed)
    df_messy_cust, df_messy_ord, injections = inject_errors(df_clean_cust, df_clean_ord, seed=seed)

    # Save individual tables
    cust_path = out_path / "customers.csv"
    ord_path = out_path / "orders.csv"
    gt_path = out_path / "ground_truth.json"

    df_messy_cust.to_csv(cust_path, index=False)
    df_messy_ord.to_csv(ord_path, index=False)

    # Also build a combined dataset with injected errors for single-table demos
    # Merge on customer_id (inner-ish representation)
    merged = df_messy_ord.merge(
        df_messy_cust[["customer_id", "name", "email", "phone", "country", "segment", "age"]],
        on="customer_id",
        how="left"
    )
    demo_path = out_path / "demo_dataset.csv"
    merged.to_csv(demo_path, index=False)

    gt_data = {
        "dataset_name": "DataFix Demo Relational Benchmark",
        "seed": seed,
        "total_injections": len(injections),
        "injections_by_type": pd.Series([inj["issue_type"] for inj in injections]).value_counts().to_dict(),
        "injections": injections
    }

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2)

    return {
        "customers_csv": str(cust_path),
        "orders_csv": str(ord_path),
        "demo_dataset_csv": str(demo_path),
        "ground_truth_json": str(gt_path),
        "total_injections": len(injections)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DataFix demo datasets with ground truth injected errors.")
    parser.add_argument("--output-dir", default="data", help="Output directory path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rows", type=int, default=5000, help="Number of rows per dataset")
    args = parser.parse_args()

    res = generate_demo_files(output_dir=args.output_dir, seed=args.seed, n_rows=args.rows)
    print(f"Generated demo datasets in {args.output_dir}:")
    print(f"  - Customers: {res['customers_csv']}")
    print(f"  - Orders: {res['orders_csv']}")
    print(f"  - Unified Demo: {res['demo_dataset_csv']}")
    print(f"  - Ground Truth: {res['ground_truth_json']} ({res['total_injections']} injected errors)")
