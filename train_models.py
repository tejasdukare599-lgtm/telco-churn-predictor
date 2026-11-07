"""
Day 2: train four classification models to predict customer churn and
compare them on the same held-out test set.

Run Day 1 first (it creates data/clean_customers.csv), then:
    python day2_train_models.py
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/clean_customers.csv"
RANDOM_STATE = 42  # fixed so you get the same numbers every run


def load_data():
    df = pd.read_csv(DATA_PATH)

    # y = what we want to predict (1 = churned, 0 = stayed)
    y = df["churn_flag"]

    # X = the information we predict from.
    # Drop the ID (useless) and the two columns that are the answer itself.
    X = df.drop(columns=["customerid", "churn", "churn_flag"])

    # Models only understand numbers, so turn text columns like
    # "Contract = Month-to-month" into 0/1 columns (one-hot encoding).
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    return X, y


def main():
    X, y = load_data()
    print(f"Rows: {len(X)}, input columns after encoding: {X.shape[1]}")

    # Hold back 20% of customers that the models never see during training.
    # stratify=y keeps the churn share the same (about 26.6%) in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # Baseline: a "model" that always says "will not churn".
    baseline_accuracy = 1 - y_test.mean()
    print(f"\nBaseline accuracy (always predict 'stays'): {baseline_accuracy:.3f}")
    print("Any real model must beat this, and accuracy alone can mislead.\n")

    # StandardScaler puts all columns on a similar scale. Logistic regression
    # and KNN need it; trees do not. Putting it inside a pipeline means it
    # learns its scaling only from the training data.
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "KNN (k=11)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11)),
        "Decision Tree (depth 5)": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)           # learn from training data
        pred = model.predict(X_test)          # predict on unseen customers
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        })

    results = pd.DataFrame(rows).sort_values("f1", ascending=False).round(3)
    print(results.to_string(index=False))

    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/model_results.csv", index=False)
    print("\nSaved reports/model_results.csv")


if __name__ == "__main__":
    main()
