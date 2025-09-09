# -*- coding: utf-8 -*-
"""
Part 4 – Discussion, Conclusion & Integration
HIT140 Foundations of Data Science (2025)
Student: Jyoti Adhikari
ID: S395089

This script integrates dataset1.csv and dataset2.csv, summarises the main findings,
and generates simple visualisations. 
It supports the final discussion and conclusion for Investigation A:
"Do bats perceive rats as potential predators?"
"""

# -----------------------------
# Import required libraries
# -----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load both datasets
# -----------------------------
df1 = pd.read_csv("/Users/jyotiadhikari/Desktop/Foundation of Data Science/Assignment2/dataset1.csv")
df2 = pd.read_csv("/Users/jyotiadhikari/Desktop/Foundation of Data Science/Assignment2/dataset2.csv")

print("Dataset1 shape:", df1.shape)
print("Dataset2 shape:", df2.shape)
print("Dataset1 columns:", df1.columns)
print("Dataset2 columns:", df2.columns)

# -----------------------------
# 2. Add rat presence variable
# -----------------------------
for df in [df1, df2]:
    # Check if the column 'rat_period_start' exists in the DataFrame
    if "rat_period_start" in df.columns:
        df["rat_presence"] = df["rat_period_start"].notna().astype(int)
    else:
        print("Warning: 'rat_period_start' column is missing in the dataset")

# -----------------------------
# 3. Combine both datasets
# -----------------------------
df = pd.concat([df1, df2], ignore_index=True)
print("Combined dataset shape:", df.shape)

# -----------------------------
# 4. Risk-taking behaviour vs rat presence
# -----------------------------
summary_risk = df.groupby("rat_presence")["risk"].mean().reset_index()
summary_risk["risk"] = summary_risk["risk"] * 100  # percentage
print("\nRisk-taking behaviour when rats present vs absent (%):")
print(summary_risk)

# -----------------------------
# 5. Reward outcome vs risk-taking behaviour
# -----------------------------
reward_risk = df.groupby("risk")["reward"].mean().reset_index()
reward_risk["reward"] = reward_risk["reward"] * 100
print("\nReward rates by risk-taking vs risk-avoidance (%):")
print(reward_risk)

# -----------------------------
# 6. Visualisations
# -----------------------------
sns.set(style="whitegrid")

# Barplot: Risk-taking behaviour with/without rats
plt.figure(figsize=(6,4))
sns.barplot(x="rat_presence", y="risk", data=df, errorbar=None)
plt.title("Risk-taking behaviour when rats present vs absent")
plt.ylabel("Proportion of risk-taking (%)")
plt.xlabel("Rat presence (0 = absent, 1 = present)")
plt.tight_layout()
plt.savefig("plot_risk_vs_rats.png")
plt.show()

# Barplot: Reward rate by risk behaviour
plt.figure(figsize=(6,4))
sns.barplot(x="risk", y="reward", data=df, errorbar=None)
plt.title("Reward rate by risk-taking vs avoidance")
plt.ylabel("Reward rate (%)")
plt.xlabel("Risk (0 = avoidance, 1 = risk-taking)")
plt.tight_layout()
plt.savefig("plot_reward_vs_risk.png")
plt.show()

# -----------------------------
# 7. Save summary results
# -----------------------------
summary_table = {
    "Risk-taking when rats absent (%)": round(summary_risk.loc[summary_risk["rat_presence"]==0, "risk"].mean() if 0 in summary_risk["rat_presence"].values else 0, 2),
    "Risk-taking when rats present (%)": round(summary_risk.loc[summary_risk["rat_presence"]==1, "risk"].mean() if 1 in summary_risk["rat_presence"].values else 0, 2),
    "Reward rate (no risk %)": round(reward_risk.loc[reward_risk["risk"]==0, "reward"].mean() if 0 in reward_risk["risk"].values else 0, 2),
    "Reward rate (risk-taking %)": round(reward_risk.loc[reward_risk["risk"]==1, "reward"].mean() if 1 in reward_risk["risk"].values else 0, 2)
}

summary_df = pd.DataFrame([summary_table])
summary_df.to_csv("final_summary_results.csv", index=False)

print("\nFinal summary table saved as final_summary_results.csv")
print("Integration completed successfully.")
