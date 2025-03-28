import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Step 1: Generate and Train Model ---
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 500

    data = {
        "monthly_income": np.random.randint(10000, 60000, n),
        "upi_txn_count": np.random.randint(5, 50, n),
        "upi_txn_amount": np.random.randint(500, 50000, n),
        "mobile_recharge": np.random.randint(50, 1000, n),
        "rent_paid_on_time": np.random.choice([0, 1], n, p=[0.3, 0.7]),
        "utility_bills_paid": np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "non_essential_spend_ratio": np.round(np.random.uniform(0.1, 0.8, n), 2),
    }

    df = pd.DataFrame(data)

    df["loan_repaid"] = (
        (df["monthly_income"] > 25000).astype(int)
        & (df["upi_txn_count"] > 10).astype(int)
        & (df["non_essential_spend_ratio"] < 0.6).astype(int)
    )

    X = df.drop("loan_repaid", axis=1)
    y = df["loan_repaid"]

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)

    return model

model = train_model()

# --- Step 2: Streamlit App Interface ---

st.set_page_config(page_title="Credit Risk Estimator", layout="centered")
st.title("🧠 AI-Based Credit Risk Estimator")
st.write("Predict the creditworthiness of users using alternative data.")

# --- Step 3: User Input Form ---
with st.form("risk_form"):
    st.subheader("Enter User Data")

    monthly_income = st.slider("Monthly Income (₹)", 5000, 100000, 25000, step=1000)
    upi_txn_count = st.slider("Monthly UPI Transaction Count", 0, 100, 20)
    upi_txn_amount = st.slider("Total UPI Txn Amount (₹)", 0, 100000, 15000, step=1000)
    mobile_recharge = st.slider("Monthly Mobile Recharge (₹)", 0, 2000, 200)
    rent_paid_on_time = st.selectbox("Rent Paid on Time?", ["Yes", "No"])
    utility_bills_paid = st.selectbox("Utility Bills Paid?", ["Yes", "No"])
    non_essential_spend_ratio = st.slider("Non-Essential Spending Ratio (0-1)", 0.0, 1.0, 0.3)

    submit = st.form_submit_button("🔍 Predict Creditworthiness")

# --- Step 4: Prediction ---
if submit:
    input_data = pd.DataFrame([{
        "monthly_income": monthly_income,
        "upi_txn_count": upi_txn_count,
        "upi_txn_amount": upi_txn_amount,
        "mobile_recharge": mobile_recharge,
        "rent_paid_on_time": 1 if rent_paid_on_time == "Yes" else 0,
        "utility_bills_paid": 1 if utility_bills_paid == "Yes" else 0,
        "non_essential_spend_ratio": non_essential_spend_ratio
    }])

    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"✅ User is **likely to repay**. (Confidence: {confidence:.2%})")
    else:
        st.error(f"⚠️ User is **at risk of default**. (Confidence: {confidence:.2%})")
