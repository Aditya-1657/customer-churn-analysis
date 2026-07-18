"""
Customer Churn Analysis - Data Cleaning & Exploratory Data Analysis
Dataset: IBM Telco Customer Churn (7,043 customers, 21 columns)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("data/telco_churn.csv")
print("Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# 2. CLEAN DATA
# ---------------------------------------------------------
# TotalCharges has blank strings for new customers (tenure=0) -> convert to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("\nMissing values before fix:\n", df.isnull().sum()[df.isnull().sum() > 0])

# Fill missing TotalCharges with 0 (these are customers with 0 tenure, i.e. brand new)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Drop customerID (not useful for analysis, just an identifier)
df.drop(columns=["customerID"], inplace=True)

# Standardize SeniorCitizen to Yes/No for readability
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# Confirm no duplicates
print("\nDuplicate rows:", df.duplicated().sum())

df.to_csv("data/telco_churn_clean.csv", index=False)
print("\nCleaned data saved to data/telco_churn_clean.csv")

# ---------------------------------------------------------
# 3. HIGH-LEVEL CHURN OVERVIEW
# ---------------------------------------------------------
churn_rate = df["Churn"].value_counts(normalize=True) * 100
print("\nChurn rate:\n", churn_rate)

plt.figure(figsize=(5, 5))
colors = ["#2E86AB", "#E63946"]
plt.pie(df["Churn"].value_counts(), labels=["Stayed", "Churned"], autopct="%1.1f%%",
        colors=colors, startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
plt.title("Overall Customer Churn Rate", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("images/01_overall_churn_rate.png")
plt.close()

# ---------------------------------------------------------
# 4. CHURN BY CONTRACT TYPE
# ---------------------------------------------------------
plt.figure(figsize=(6, 4.5))
ct = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
ct.plot(kind="bar", stacked=True, color=["#2E86AB", "#E63946"], ax=plt.gca())
plt.title("Churn Rate by Contract Type", fontsize=13, fontweight="bold")
plt.ylabel("Percentage of Customers")
plt.xlabel("Contract Type")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.savefig("images/02_churn_by_contract.png")
plt.close()

# ---------------------------------------------------------
# 5. CHURN BY TENURE (how long they've stayed)
# ---------------------------------------------------------
plt.figure(figsize=(7, 4.5))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack",
             palette=["#2E86AB", "#E63946"], bins=30)
plt.title("Customer Tenure Distribution by Churn Status", fontsize=13, fontweight="bold")
plt.xlabel("Tenure (months)")
plt.tight_layout()
plt.savefig("images/03_tenure_distribution.png")
plt.close()

# ---------------------------------------------------------
# 6. CHURN BY MONTHLY CHARGES
# ---------------------------------------------------------
plt.figure(figsize=(7, 4.5))
sns.kdeplot(data=df[df.Churn == "No"], x="MonthlyCharges", fill=True, color="#2E86AB", label="Stayed", alpha=0.5)
sns.kdeplot(data=df[df.Churn == "Yes"], x="MonthlyCharges", fill=True, color="#E63946", label="Churned", alpha=0.5)
plt.title("Monthly Charges Distribution by Churn Status", fontsize=13, fontweight="bold")
plt.xlabel("Monthly Charges ($)")
plt.legend()
plt.tight_layout()
plt.savefig("images/04_monthly_charges.png")
plt.close()

# ---------------------------------------------------------
# 7. CHURN BY INTERNET SERVICE TYPE
# ---------------------------------------------------------
plt.figure(figsize=(6, 4.5))
ct2 = pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100
ct2.plot(kind="bar", stacked=True, color=["#2E86AB", "#E63946"], ax=plt.gca())
plt.title("Churn Rate by Internet Service Type", fontsize=13, fontweight="bold")
plt.ylabel("Percentage of Customers")
plt.xlabel("Internet Service")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.savefig("images/05_churn_by_internet_service.png")
plt.close()

# ---------------------------------------------------------
# 8. CHURN BY PAYMENT METHOD
# ---------------------------------------------------------
plt.figure(figsize=(7, 4.5))
ct3 = pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index") * 100
ct3.sort_values("Yes", ascending=False).plot(kind="barh", stacked=True,
                                              color=["#2E86AB", "#E63946"], ax=plt.gca())
plt.title("Churn Rate by Payment Method", fontsize=13, fontweight="bold")
plt.xlabel("Percentage of Customers")
plt.legend(title="Churn")
plt.tight_layout()
plt.savefig("images/06_churn_by_payment_method.png")
plt.close()

print("\nAll EDA charts saved to images/ folder.")

# ---------------------------------------------------------
# 9. KEY NUMBERS FOR THE REPORT / README
# ---------------------------------------------------------
summary = {
    "Total customers": len(df),
    "Churned customers": (df.Churn == "Yes").sum(),
    "Overall churn rate (%)": round(churn_rate["Yes"], 2),
    "Month-to-month churn rate (%)": round(ct.loc["Month-to-month", "Yes"], 2),
    "Two year contract churn rate (%)": round(ct.loc["Two year", "Yes"], 2),
    "Fiber optic churn rate (%)": round(ct2.loc["Fiber optic", "Yes"], 2),
    "Avg tenure - churned (months)": round(df[df.Churn == "Yes"].tenure.mean(), 1),
    "Avg tenure - stayed (months)": round(df[df.Churn == "No"].tenure.mean(), 1),
    "Avg monthly charges - churned ($)": round(df[df.Churn == "Yes"].MonthlyCharges.mean(), 2),
    "Avg monthly charges - stayed ($)": round(df[df.Churn == "No"].MonthlyCharges.mean(), 2),
}
print("\n=== KEY INSIGHTS ===")
for k, v in summary.items():
    print(f"{k}: {v}")

with open("key_insights.txt", "w") as f:
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")
