# 📊 Netflix Customer Churn Prediction Dashboard

This project builds and deploys an interactive dashboard to predict customer churn for a Netflix-like subscription service. It compares logistic regression and random forest models to determine the best predictor of churn and identifies the top 5 riskiest customers based on monthly value.

## 🚀 Features

- Compares two ML models: Logistic Regression & Random Forest
- Displays ROC AUC to determine best model
- Predicts customer churn probabilities
- Highlights top 5 high-value at-risk customers
- Visualizes churn distribution
- Outputs classification report and risk charts

## 🧠 ML Workflow

1. Load and clean dataset
2. One-hot encode categorical variables
3. Split data into train/test sets
4. Train Logistic Regression and Random Forest
5. Compare models using ROC AUC
6. Use best model for churn prediction
7. Visualize and explain results in Streamlit dashboard

## 📁 Files

- `app.py`: Streamlit app (main dashboard)
- `netflix_customer_churn.csv`: Dataset file
- `requirements.txt`: Dependencies for running the app

## 📦 Installation

```bash
git clone https://github.com/farihanuzhat26/customer-churn-netflix.git
cd customer-churn-netflix
pip install -r requirements.txt
streamlit run app.py
