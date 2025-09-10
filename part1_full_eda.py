"""
Part 1
Assessment: HIT140- Investigation A
Script name: part1_full_eda.py

This script generates:
- CSV tables and PNG visualizations (saved in outputs/directory)
- An optional cleaned dataset

Author: Member 1 (Ruhi Rayamajhi- S396926)
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Config
DATASET1_PATH = "dataset1.csv"    #required
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helpers functions
def safe_read_csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}")
        return None
    
def safe_to_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.to_datetime(series.astype(str), errors="coerce")
        
def save_table(df, name, index=True):
    out = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(out, index=index)
    print(f"[saved] {out}")
    return out 
    
def save_fig(fig, name):
    out = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")
    return out 
    
def counts_with_pct(series):
    vc = series.value_counts(dropna=False).sort_index().to_frame("count")
    vc["pct"] = (vc["count"] / len(series)*100).round(2)
    return vc

# load data
print("Loading dataset1...")
df1 = safe_read_csv(DATASET1_PATH)
if df1 is None:
    print(f"ERROR: {DATASET1_PATH} not found. Put dataset1.csv in the script folder and re-run.")
    sys.exit(1)
print(f"dataset1 shape: {df1.shape}")
print("Column:", list(df1.columns))

# Basic Cleaning & Data Type Fixes
# Step 1: Parse columns that are likely datetime
for col in df1.columns:
    if "time" in col.lower() or "date" in col.lower():
        df1[col] = safe_to_datetime(df1[col])

# Step 2: Convert risk/reward columns to nullable integers (if available)
for col in ["risk", "reward"]:
    if col in df1.columns:
        df1[col] = pd.to_numeric(df1[col], errors="coerce").astype("Int64")

# Step 3: Standardize categorical columns (if available)
for cat_col in ["habit", "month", "season"]:
    if cat_col in df1.columns:
        df1[cat_col] = df1[cat_col].astype("category")

# Step 4: Create 'rat_present_at_landing' using multiple fallbacks
# a) First check seconds_after_rat_arrival (>=0 means rat present)
# b) If that's missing, compare start_time against rat_period_start/rat_period_end
# c) If still not available, just leave as NaN
if "seconds_after_rat_arrival" in df1.columns:
    # Rule: keep NaN if original is NaN, set True if value >= 0, set False if value < 0
    s = pd.to_numeric(df1["seconds_after_rat_arrival"], errors="coerce")
    df1["rat_present_at_landing"] = pd.Series(np.where(s.isna(), pd.NA, s >= 0), index=df1.index).astype("object")
    # Convert any "True"/"False" string value into proper booleans (or NA if missing)
    df1["rat_present_at_landing"] = df1["rat_present_at_landing"].apply(lambda x: True if x is True else (False if x is False else pd.NA))
    print("[info] Derived rat_present_at_landing from seconds_after_rat_arrival.")
elif set(["rat_period_start", "rat_period_end", "start_time"]).issubset(df1.columns):
    # if all required columns are present: parse time and comprae
    rp_s = safe_to_datetime(df1["rat_period_start"])
    rp_e = safe_to_datetime(df1["rat_period_end"])
    st = safe_to_datetime(df1["start_time"])
    df1["rat_present_at_landing"] = pd.Series(pd.NA, index=df1.index)
    mask_valid = rp_s.notna() & rp_e.notna() & st.notna()
    df1.loc[mask_valid, "rat_present_at_landing"] = ((st >= rp_s) & (st <= rp_e))
    print("[info] Derived rat_present_at_landing from rat_period_start/end and start_time.")
else:
    df1["rat_present_at_landing"] = pd.NA
    print("[warn] No direct rat timing column found - rat_present_at_landing set to NA for all rows. If seconds_after_rat_arrival exists but with strange format, check raw data.")

# Step 5: Basic duplicate check
dups = df1.duplicated().sum()
if dups > 0:
    print(f"[warn] Found {dups} duplicated rows in dataset1 (consider investigating).")

# Data quality summary
missing = df1.isna().sum().sort_values(ascending=False).to_frame("missing_count")
missing["missing_pct"] = (missing["missing_count"] / len(df1) * 100).round(2)
print("\nMissingness summary (top 20):")
save_table(missing, "part1_missingness_summary")

numeric_cols = df1.select_dtypes(include=[np.number]).columns.to_list()
numeric_summary = df1[numeric_cols].describe().T.round(3) if numeric_cols else pd.DataFrame()
if not numeric_summary.empty:
    print("\nNumeric summary (describe):")
    print(numeric_summary)
    save_table(numeric_summary, "part1_numeric_summary")

# Counts & crossrtab analysis
# Key categorical counts
for col in["habit", "month", "season", "risk", "reward", "rat_present_at_landing"]:
    if col in df1.columns:
        table = counts_with_pct(df1[col])
        print(f"\nCounts for {col}:")
        print(table)
        save_table(table, f"part1_counts_{col}")

# Cross tabs of interest
if "rat_present_at_landing" in df1.columns and "risk" in df1.columns:
    ct = pd.crosstab(df1["rat_present_at_landing"], df1["risk"], dropna=False)
    ct_pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(2)
    print("\nCrosstab: rat_present_at_landing vs risk (counts):")
    print(ct)
    print("\nCrosstab (row %):")
    print(ct_pct)
    save_table(ct, "part1_rat_present_vs_risk_count")
    save_table(ct_pct, "part1_rat_present_vs_risk_rowpct")

# Chi-square test if the table is roughly 2x2
try:
    # make a numeric contigency table (drop NA rows/columns)
    ct2 = ct.copy()
    # remove any rows or columns that are NA, if they exist
    ct2 = ct2.loc[[r for r in ct2.index if pd.notna(r)], [c for c in ct2.columns if pd.notna(c)]]
    if ct2.shape[0] >= 2 and ct2.shape[1] >= 2 and ct2.values.sum() > 0:
        chi2, p, dof, expected = chi2_contingency(ct2)
        print("f\nChi-square test: chi2={chi2:.3f}, p-value={p:.4f}, dof={dof}")
        exp_df = pd.DataFrame(expected, index=ct2.index, columns=ct2.columns).round(2)
        save_table(exp_df, "part1_chi2_expected_counts")
    else: 
        print("[info] Not enough valid cells for chi-square test.")
except Exception as e:
    print("[warn] Chi-square test failed:", e)

if "rat_present_at_landing" in df1.columns and "reward" in df1.columns:
    ct2 = pd.crosstab(df1["rat_present_at_landing"], df1["reward"], dropna=False)
    ct2_pct = (ct2.div(ct2.sum(axis=1), axis=0) * 100).round(2)
    save_table(ct2, "part1_rat_present_vs_reward_count")
    save_table(ct2_pct, "part1_rat_present_vs_reward_rowpct")

# plots (matplotlib only)
# Each plot will be saved as a PNG in the outputs/ folder
# 1) Histofram of seconds_after_rat_arrival
if "seconds_after_rat_arrival" in df1.columns:
    fig = plt.figure()
    series = pd.to_numeric(df1["seconds_after_rat_arrival"], errors="coerce").dropna()
    if len(series) > 0:
        series.plot(kind="hist", bins=30, title="Distribution: seconds_after_rat_arrival")
        plt.xlabel("seconds_after_rat_arrival")
        plt.ylabel("frequency")
        save_fig(fig, "hist_seconds_after_rat_arrival")
    else:
        plt.close(fig)

# 2) Bar: risk
if "risk" in df1.columns:
    fig = plt.figure()
    counts = df1["risk"].value_counts(dropna=False).sort_index()
    counts.plot(kind="bar", title="Risk-taking behaviour counts (0=avoid, 1=risk)")
    plt.xlabel("risk")
    plt.ylabel("count")
    save_fig(fig, "bar_risk_counts")

# 3) Bar: reward
if "reward" in df1.columns:
    fig = plt.figure()
    counts = df1["reward"].value_counts(dropna=False).sort_index()
    counts.plot(kind="bar", title="Reward outcome counts (0=no, 1=yes)")
    plt.xlabel("reward")
    plt.ylabel("count")
    save_fig(fig, "bar_reward_counts")

# 4) Bar: rat_present_at_landing
if "rat_present_at_landing" in df1.columns:
    fig = plt.figure()
    counts = df1["rat_present_at_landing"].value_counts(dropna=False).sort_index()
    counts.plot(kind="bar", title="Rat present at landing (derived)")
    plt.xlabel("rat_present_at_landing")
    plt.ylabel("count")
    save_fig(fig, "bar_rat_present_counts")

# 5) Boxplot: bat_landing_to_food by rat_present_at_landing
if "bat_landing_to_food" in df1.columns and "rat_present_at_landing" in df1.columns:
    data_no = df1.loc[df1["rat_present_at_landing"] == False, "bat_landing_to_food"].dropna()
    data_yes = df1.loc[df1["rat_present_at_landing"] == True, "bat_landing_to_food"].dropna()
    if len(data_no)>0 or len(data_yes)>0:
        fig = plt.figure()
        plt.boxplot([data_no, data_yes], labels=["No rats present", "Rats present"])
        plt.title("Time from landing to approaching food, by rat presence")
        plt.ylabel("seconds (bat_landing_to_food)")
        save_fig(fig, "box_bat_landing_to_food_by_rat_presence")

# 6) Scatter: bat_landing_to_food vs seconds_after_rat_arrival (if both numeric)
if "bat_landing_to_food" in df1.columns and "seconds_after_rat_arrival" in df1.columns:
    x = pd.to_numeric(df1["seconds_after_rat_arrival"], errors="coerce")
    y = pd.to_numeric(df1["bat_landing_to_food"], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() > 0:
        fig = plt.figure()
        plt.scatter(x[mask], y[mask], s=8)
        plt.title("bat_landing_to_food vs seconds_after_rat_arrival")
        plt.xlabel("seconds_after_rat_arrival")
        plt.ylabel("bat_landing_to_food (sec)")
        save_fig(fig, "scatter_landing_to_food_vs_seconds_after_rat")

# 7) Simple time aggreration: count of landings per month (if month exists)
if "month" in df1.columns:
    try:
        monthly = df1["month"].value_counts().sort_index()
        fig = plt.figure()
        monthly.plot(kind="bar", title="Bat landing counts by month")
        plt.xlabel("month")
        plt.ylabel("landings")
        save_fig(fig, "bar_landings_by_month")
    except Exception as e:
        print("[warn] Could not create monthly plot:", e)

# save cleaned data
clean_out = os.path.join(OUTPUT_DIR, "dataset1_cleaned_for_part1.csv")
df1.to_csv(clean_out, index=False)
print(f"[saved] cleaned dataset -> {clean_out}")

print("\nDONE. Check the 'outputs' folder for CSVs and PNGs to paste into slides.")