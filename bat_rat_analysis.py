# ============================================================
# Bat vs Rat: Full Analysis Pipeline
# HIT140 Foundations of Data Science Group 65
# Files: dataset1.csv, dataset2.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# ---------------- Display Settings ----------------
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 50)

# ---------------- File Paths & Output Directory ----------------
DATA1_PATH = "dataset1.csv"
DATA2_PATH = "dataset2.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------- Helper Functions ----------------
def safe_mode(series: pd.Series):
    m = series.mode(dropna=True)
    return m.iloc[0] if len(m) else np.nan

def print_section(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def classify_skewness(skew_val: float, tol: float = 0.1):
    if np.isnan(skew_val):
        return "unknown"
    if skew_val > tol:
        return "right-skewed"
    if skew_val < -tol:
        return "left-skewed"
    return "approximately symmetric"

# ---------------- Data Preparation & Descriptive Statistics ----------------
def descriptives(df1: pd.DataFrame, df2: pd.DataFrame):
    print_section("Data Preparation & Descriptive Statistics")

    print("Dataset 1 Shape:", df1.shape)
    print("Dataset 2 Shape:", df2.shape)

    df1c = df1.copy()
    df2c = df2.copy()

    numeric_like_df1 = ["bat_landing_to_food", "seconds_after_rat_arrival", "hours_after_sunset", "risk", "reward"]
    numeric_like_df2 = ["hours_after_sunset", "bat_landing_number", "food_availability",
                        "rat_minutes", "rat_arrival_number"]

    for col in numeric_like_df1:
        if col in df1c.columns:
            df1c[col] = pd.to_numeric(df1c[col], errors="coerce")
    for col in numeric_like_df2:
        if col in df2c.columns:
            df2c[col] = pd.to_numeric(df2c[col], errors="coerce")

    key_cols1 = [c for c in numeric_like_df1 if c in df1c.columns]
    key_cols2 = [c for c in numeric_like_df2 if c in df2c.columns]
    df1c = df1c.dropna(subset=key_cols1, how="all")
    df2c = df2c.dropna(subset=key_cols2, how="all")

    quant_cols = [c for c in ["bat_landing_to_food", "seconds_after_rat_arrival", "hours_after_sunset"] if c in df1c.columns]

    print("\nDescriptive statistics (Dataset 1):")
    for col in quant_cols:
        s = df1c[col].dropna()
        mean_ = s.mean()
        median_ = s.median()
        mode_ = safe_mode(s)
        std_ = s.std()
        var_ = s.var()
        data_range = s.max() - s.min()
        iqr_ = s.quantile(0.75) - s.quantile(0.25)
        skew_ = s.skew()
        shape_ = classify_skewness(skew_)

        print(f"\n--- {col} ---")
        print(f"Mean: {mean_:.3f}")
        print(f"Median: {median_:.3f}")
        print(f"Mode: {mode_:.3f}" if pd.notna(mode_) else "Mode: NaN")
        print(f"Std Dev: {std_:.3f}")
        print(f"Variance: {var_:.3f}")
        print(f"Range: {data_range:.3f}")
        print(f"IQR: {iqr_:.3f}")
        print(f"Skewness: {skew_:.3f} → {shape_}")

        plt.figure(figsize=(7,4))
        sns.histplot(s, bins=20, kde=True)
        plt.title(f"Histogram — {col}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"hist_{col}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6,3.5))
        sns.boxplot(x=s)
        plt.title(f"Boxplot — {col}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"box_{col}.png", dpi=200)
        plt.close()

    return df1c, df2c

# ---------------- Qualitative Analysis ----------------
def qualitative_analysis(df1c: pd.DataFrame):
    print_section("Qualitative Analysis & Frequency Tables")
    candidate_cats = ["habit", "risk", "reward", "season", "month"]
    cat_cols = [c for c in candidate_cats if c in df1c.columns]

    for col in cat_cols:
        counts = df1c[col].astype("category").value_counts(dropna=False)
        print(f"\nFrequency table — {col}:\n{counts}")

        plt.figure(figsize=(7,4))
        counts.plot(kind="bar")
        plt.title(f"Bar Chart — {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"bar_{col}.png", dpi=200)
        plt.close()

        if counts.shape[0] <= 8:
            plt.figure(figsize=(5.5,5.5))
            counts.plot(kind="pie", autopct="%1.1f%%", ylabel="")
            plt.title(f"Pie Chart — {col}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"pie_{col}.png", dpi=200)
            plt.close()

# ---------------- Inferential Statistics ----------------
def inferential_stats(df1c: pd.DataFrame):
    print_section("Inferential Statistics")

    if "bat_landing_to_food" in df1c.columns:
        data = pd.to_numeric(df1c["bat_landing_to_food"], errors="coerce").dropna()
        if len(data) >= 3:
            conf = 0.95
            mean_ = np.mean(data)
            sem_ = stats.sem(data)
            ci_low, ci_high = stats.t.interval(conf, len(data)-1, loc=mean_, scale=sem_)
            print(f"95% CI for mean bat_landing_to_food: ({ci_low:.3f}, {ci_high:.3f}) | mean={mean_:.3f}, n={len(data)}")

    if "seconds_after_rat_arrival" in df1c.columns:
        test_data = pd.to_numeric(df1c["seconds_after_rat_arrival"], errors="coerce").dropna()
        if len(test_data) >= 3:
            t_stat, p_two_sided = stats.ttest_1samp(test_data, 0.0)
            mean_test = test_data.mean()
            p_one_sided = p_two_sided / 2 if mean_test > 0 else 1 - (p_two_sided / 2)
            direction = "greater than 0" if mean_test > 0 else "NOT greater than 0"
            print(f"\nOne-sample t-test on seconds_after_rat_arrival vs 0:")
            print(f"t={t_stat:.3f}, two-sided p={p_two_sided:.4f}, one-sided p={p_one_sided:.4f} → mean is {direction} (mean={mean_test:.3f}, n={len(test_data)})")

    if all(c in df1c.columns for c in ["risk", "reward"]):
        risk = pd.to_numeric(df1c["risk"], errors="coerce")
        reward = pd.to_numeric(df1c["reward"], errors="coerce")
        ct = pd.crosstab(risk, reward)
        if ct.shape == (2,2):
            chi2, p, dof, exp = stats.chi2_contingency(ct)
            print("\nChi-square test of independence — Risk vs Reward")
            print("Contingency Table:\n", ct)
            print(f"chi2={chi2:.3f}, dof={dof}, p-value={p:.4f}")
            print("Significant association" if p < 0.05 else "No significant association")
        else:
            print("\nRisk/Reward contingency table (not strictly binary):\n", ct)

# ---------------- Feature Engineering & Scatter Plots ----------------
def parse_time_cols_and_ratstay(df1c: pd.DataFrame):
    if "rat_period_start" in df1c.columns and "rat_period_end" in df1c.columns:
        start = pd.to_datetime(df1c["rat_period_start"], errors="coerce", dayfirst=True)
        end   = pd.to_datetime(df1c["rat_period_end"], errors="coerce", dayfirst=True)

        df1c["RatStay"] = (end - start).dt.total_seconds()
    else:
        df1c["RatStay"] = np.nan
    return df1c

def features_and_plots(df1c: pd.DataFrame, df2c: pd.DataFrame):
    print_section("Feature Engineering & Scatter Plots")
    df1c = parse_time_cols_and_ratstay(df1c)

    if "rat_minutes" in df2c.columns:
        df2c["RatIntensity"] = pd.to_numeric(df2c["rat_minutes"], errors="coerce") / 30.0

    if all(c in df1c.columns for c in ["risk", "reward"]):
        temp = df1c[["risk", "reward"]].copy()
        temp["risk_num"] = pd.to_numeric(temp["risk"], errors="coerce")
        temp["reward_num"] = pd.to_numeric(temp["reward"], errors="coerce")
        x = temp["risk_num"] + np.random.uniform(-0.05, 0.05, size=len(temp))
        y = temp["reward_num"] + np.random.uniform(-0.05, 0.05, size=len(temp))
        plt.figure(figsize=(6.5,4.5))
        plt.scatter(x, y, alpha=0.6)
        plt.title("Scatter: Risk vs Reward (with jitter)")
        plt.xlabel("Risk (0/1)")
        plt.ylabel("Reward (0/1)")
        plt.yticks([0,1])
        plt.xticks([0,1])
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "scatter_risk_reward.png", dpi=200)
        plt.close()

    if all(c in df1c.columns for c in ["hours_after_sunset", "bat_landing_to_food"]):
        plt.figure(figsize=(6.5,4.5))
        sns.scatterplot(data=df1c, x="hours_after_sunset", y="bat_landing_to_food", alpha=0.7)
        plt.title("Scatter: Hours After Sunset vs Bat Landing to Food")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "scatter_hours_vs_landing.png", dpi=200)
        plt.close()

    if all(c in df1c.columns for c in ["RatStay", "reward"]):
        plt.figure(figsize=(6.5,4.5))
        sns.scatterplot(data=df1c, x="RatStay", y=pd.to_numeric(df1c["reward"], errors="coerce"), alpha=0.7)
        plt.title("Scatter: RatStay (sec) vs Reward")
        plt.ylabel("Reward (0/1)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "scatter_ratstay_reward.png", dpi=200)
        plt.close()

    df1c.to_csv(OUTPUT_DIR / "dataset1_processed.csv", index=False)
    df2c.to_csv(OUTPUT_DIR / "dataset2_processed.csv", index=False)
    print("\nSaved processed CSVs and all figures in 'outputs/' folder.")

# ---------------- Main Execution ----------------
def main():
    df1 = pd.read_csv(DATA1_PATH)
    df2 = pd.read_csv(DATA2_PATH)

    df1c, df2c = descriptives(df1, df2)
    qualitative_analysis(df1c)
    inferential_stats(df1c)
    features_and_plots(df1c, df2c)

if __name__ == "__main__":
    main()
