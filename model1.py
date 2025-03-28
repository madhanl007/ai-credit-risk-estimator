import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_and_save_model():
    # Simulated training data
    data = {
        'monthly_income': [25000, 18000, 45000, 32000, 15000],
        'monthly_expense': [12000, 10000, 22000, 15000, 8000],
        'upi_transactions_per_month': [40, 20, 60, 35, 15],
        'rent_paid': [8000, 5000, 12000, 10000, 4000],
        'mobile_recharge_amt': [300, 200, 500, 350, 150],
        'label': [1, 0, 1, 1, 0]  # 1 = good credit, 0 = risky
    }

    df = pd.DataFrame(data)

    X = df.drop('label', axis=1)
    y = df['label']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model to file
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved as model.pkl")

if __name__ == "__main__":
    train_and_save_model()
