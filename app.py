echo "# customer-churn-netflix" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/farihanuzhat26/customer-churn-netflix.git
git push -u origin main

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# === Load dataset ===
df = pd.read_csv("netflix_customer_churn.csv")

# === Preserve original columns ===
customer_ids_all = df["customer_id"]
monthly_fees_all = df["monthly_fee"]

# === One-hot encoding ===
df_encoded = pd.get_dummies(df.drop(columns=["customer_id"]), drop_first=True)

# === Define X and y ===
X = df_encoded.drop("churned", axis=1)
y = df_encoded["churned"]

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === Train Logistic Regression ===
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_prob)

# === Train Random Forest ===
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_prob)

# === Choose best model ===
if rf_auc > lr_auc:
    best_model_name = "Random Forest"
    y_pred = rf_pred
    y_prob = rf_prob
    model_used = rf_model
else:
    best_model_name = "Logistic Regression"
    y_pred = lr_pred
    y_prob = lr_prob
    model_used = lr_model

# === Prepare risk table ===
customer_ids = customer_ids_all.iloc[X_test.index].values
monthly_fees = monthly_fees_all.iloc[X_test.index].values

risk_df = X_test.copy()
risk_df["customer_id"] = customer_ids
risk_df["monthly_fee"] = monthly_fees
risk_df["churn_risk"] = y_prob
risk_df["Actual_Churn"] = y_test.values

at_risk = risk_df[risk_df["churn_risk"] >= 0.5]
top5 = at_risk.sort_values(by="monthly_fee", ascending=False).head(5)

# === Streamlit Dashboard ===
st.title("📊 Netflix Customer Churn Prediction")

# Metrics section
st.subheader("📈 Model Comparison (on Test Set)")
st.metric("Logistic Regression AUC", f"{lr_auc:.4f}")
st.metric("Random Forest AUC", f"{rf_auc:.4f}")
st.success(f"✅ Using Best Model: {best_model_name}")

# Churn distribution
st.subheader("Churn Distribution")
fig1, ax1 = plt.subplots()
y_test.value_counts().plot.pie(labels=["Stayed", "Churned"], autopct='%1.1f%%',
                               colors=["lightgreen", "salmon"], ax=ax1)
ax1.set_ylabel("")
st.pyplot(fig1)

# Top 5 at-risk
st.subheader("Top 5 Riskiest Customers (by Monthly Fee)")
st.dataframe(top5[["customer_id", "monthly_fee", "churn_risk", "Actual_Churn"]])

# Risk bar chart
st.subheader("Risk Scores of Top 5")
fig2, ax2 = plt.subplots()
sns.barplot(x="churn_risk", y="customer_id", data=top5, palette="Reds_r", ax=ax2)
plt.xlabel("Churn Probability")
plt.ylabel("Customer ID")
st.pyplot(fig2)

# Classification report
st.subheader(f"{best_model_name} - Classification Report")
st.text(classification_report(y_test, y_pred))
