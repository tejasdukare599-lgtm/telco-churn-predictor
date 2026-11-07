# Telco Customer Churn Predictor

Predicting which telecom customers are about to leave, using SQL for exploration and four classical machine learning models for prediction.

I built this to get hands-on practice with the full classical ML workflow on a business problem: explore data with SQL, train and compare several algorithms fairly, tune them, and judge them with the metrics that actually matter for churn (recall and F1, not just accuracy).

## Dataset

[Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d), a public IBM sample dataset: 7,043 customers, 21 columns (contract type, tenure, services, billing, charges, and whether the customer churned). 11 rows with a blank `TotalCharges` (brand-new customers) were dropped, leaving 7,032. About 26.6% of customers churned.

## Approach

1. **Load and clean** the CSV with Pandas and store it in a SQLite database.
2. **Explore with SQL** (6 queries in `sql/queries.sql`) to find which customer groups churn most.
3. **Train four models** on a stratified 80/20 train/test split: logistic regression, KNN, decision tree, random forest. Scaling is done inside a pipeline so test data never leaks into training.
4. **Cross-validate** (5-fold) on the training data, then **tune** logistic regression and random forest with grid search.
5. **Evaluate** the tuned models on the untouched test set with a confusion matrix and feature chart.

## Key findings from SQL

| Group | Churn rate |
|---|---|
| Month-to-month contract | 42.7% |
| One-year contract | 11.3% |
| Two-year contract | 2.8% |
| First 12 months of tenure | 47.7% |
| 49+ months of tenure | 9.5% |
| Fiber optic internet | 41.9% |
| DSL internet | 19.0% |
| Pays by electronic check | 45.3% |

Customers who churned also paid more per month on average (74.44 vs 61.31) and had far shorter tenure (18.0 vs 37.7 months).

## Model results

A model that always predicts "customer stays" is already right **73.4%** of the time, so accuracy alone is misleading here. I focus on **recall** (how many churners are caught) and **F1**.

**Default settings, test set:**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.648 | 0.575 | 0.609 |
| Decision Tree (depth 5) | 0.778 | 0.581 | 0.596 | 0.588 |
| Random Forest | 0.787 | 0.621 | 0.508 | 0.559 |
| KNN (k=11) | 0.770 | 0.569 | 0.548 | 0.559 |

**5-fold cross-validation (F1, training data):** Logistic Regression 0.596, KNN 0.568, Random Forest 0.561, Decision Tree 0.551.

**After tuning (test set):**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest (tuned) | 0.756 | 0.527 | 0.791 | 0.632 |
| Logistic Regression (tuned) | 0.726 | 0.491 | 0.797 | 0.608 |

The tuned random forest uses `class_weight="balanced"`, `max_depth=None`, `min_samples_leaf=5`. Telling the model that missing a churner costs more than a false alarm raised recall from about 0.51 to 0.79. The trade-off is lower accuracy (0.756, only slightly above the 0.734 baseline) and more false alarms.

![Confusion matrix](reports/confusion_matrix.png)

The best model catches 296 of 374 churners (78 missed) and wrongly flags 266 customers who stayed.

![Feature chart](reports/feature_importance.png)

Short tenure, month-to-month contracts and fiber optic service are the clearest signals of churn.

## Limitations

- **Correlated features.** Monthly charges, total charges, tenure and internet type move together, so individual logistic regression coefficients are unreliable. For example, `monthlycharges` appears to protect against churn in the chart even though churners pay more on average in the raw data. Read the chart for the strongest signals only.
- **Model choice used the test set.** I picked the better of the two tuned models by its test score. A cleaner setup would use a separate validation set.
- **One public sample dataset.** Results may not carry over to a real telecom's data.
- **Default decision threshold.** I did not tune the probability cutoff, and I did not model the real business cost of a missed churner versus a wasted retention offer.

## Possible next steps

Tune the decision threshold against a cost estimate, try gradient boosting (XGBoost / LightGBM), handle imbalance with resampling, and add a validation set.

## How to run

```bash
pip install -r requirements.txt
python day1_load_and_clean.py data/Telco-Customer-Churn.csv   # cleans data, builds SQLite DB, runs SQL queries
python day2_train_models.py                                   # trains and compares four models
python day3_evaluate_and_tune.py                              # cross-validation, tuning, charts (about 1 minute)
```

Results and charts are saved to the `reports/` folder.

## Project structure

```
data/Telco-Customer-Churn.csv     raw dataset
sql/queries.sql                   SQL exploration queries
day1_load_and_clean.py            clean data, create SQLite database, run queries
day2_train_models.py              train and compare four classifiers
day3_evaluate_and_tune.py         cross-validation, tuning, confusion matrix, feature chart
reports/                          result tables and charts
```

## Tech

Python, Pandas, scikit-learn, SQLite, Matplotlib

**Author:** Tejas Dukare, [LinkedIn](https://www.linkedin.com/in/tejas-dukare-141161328)
