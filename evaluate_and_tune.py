"""
Day 3: check the models more carefully (cross-validation), tune the two most
promising ones, then draw a confusion matrix and a feature-importance chart.

Run Day 1 and Day 2 first, then:
    python day3_evaluate_and_tune.py
This takes about a minute.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/clean_customers.csv"
RANDOM_STATE = 42
REPORTS = Path("reports")


def load_data():
    df = pd.read_csv(DATA_PATH)
    y = df["churn_flag"]
    X = df.drop(columns=["customerid", "churn", "churn_flag"])
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    return X, y


def score(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }


def main():
    REPORTS.mkdir(exist_ok=True)
    X, y = load_data()
    # Same split as Day 2, so results are comparable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---- Part 1: cross-validation (uses ONLY the training data) -----------
    # The training data is cut into 5 parts. Each part takes a turn as the
    # mini test set. This shows how stable each model is, not one lucky split.
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "KNN (k=11)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11)),
        "Decision Tree (depth 5)": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }
    cv_rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        cv_rows.append({"model": name, "cv_f1_mean": scores.mean(), "cv_f1_std": scores.std()})
    cv_results = pd.DataFrame(cv_rows).sort_values("cv_f1_mean", ascending=False).round(3)
    print("5-fold cross-validation (F1 on training data):")
    print(cv_results.to_string(index=False))
    cv_results.to_csv(REPORTS / "cv_results.csv", index=False)

    # ---- Part 2: tuning --------------------------------------------------
    # Try several settings and keep the best one (judged by cross-validation,
    # still without touching the test set).
    # class_weight="balanced" tells the model that missing a churner is
    # costly, which usually raises recall.
    searches = {
        "Logistic Regression (tuned)": GridSearchCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            {"logisticregression__C": [0.01, 0.1, 1, 10],
             "logisticregression__class_weight": [None, "balanced"]},
            scoring="f1", cv=cv, n_jobs=-1),
        "Random Forest (tuned)": GridSearchCV(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            {"max_depth": [5, 8, None],
             "min_samples_leaf": [1, 5, 10],
             "class_weight": [None, "balanced"]},
            scoring="f1", cv=cv, n_jobs=-1),
    }
    tuned_rows = []
    best_models = {}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        print(f"\n{name}: best settings = {search.best_params_}")
        best_models[name] = search.best_estimator_
        tuned_rows.append({"model": name, **score(search.best_estimator_, X_test, y_test)})

    tuned = pd.DataFrame(tuned_rows).sort_values("f1", ascending=False).round(3)
    print("\nTuned models on the held-out test set:")
    print(tuned.to_string(index=False))
    tuned.to_csv(REPORTS / "tuned_results.csv", index=False)

    # ---- Part 3: look closely at the best tuned model --------------------
    best_name = tuned.iloc[0]["model"]
    best = best_models[best_name]
    print(f"\nBest tuned model: {best_name}")
    print(classification_report(y_test, best.predict(X_test),
                                target_names=["stayed", "churned"], digits=3))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        best, X_test, y_test, display_labels=["stayed", "churned"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix: {best_name}")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ---- Part 4: which features matter? ----------------------------------
    # For logistic regression the scaled coefficients show direction and size:
    # positive = pushes toward churn, negative = pushes toward staying.
    lr = best_models["Logistic Regression (tuned)"]
    coefs = pd.Series(lr.named_steps["logisticregression"].coef_[0], index=X.columns)
    top = pd.concat([coefs.nsmallest(7), coefs.nlargest(7)]).sort_values()
    print("\nStrongest features (logistic regression coefficients):")
    print(top.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    top.plot.barh(ax=ax, color=["#2a9d8f" if v < 0 else "#e76f51" for v in top])
    ax.set_title("Toward leaving (red) or staying (green)", fontsize=11)
    ax.set_xlabel("Logistic regression coefficient (scaled)")
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved results and charts to the {REPORTS}/ folder.")


if __name__ == "__main__":
    main()
