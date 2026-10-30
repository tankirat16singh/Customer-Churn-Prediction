
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# 1. Load the dataset
data = pd.read_csv("Telco-Customer-Churn.csv")

# 2. Clean the TotalCharges column
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)

# 3. Remove customer ID because it is not a useful predictor
data = data.drop(columns=["customerID"])

# 4. Separate inputs and target
X = data.drop(columns=["Churn"])
y = data["Churn"].map({"No": 0, "Yes": 1})

# 5. Identify numeric and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# 6. Prepare preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 7. Create the complete ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    ))
])

# 8. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Train the model
print("Training model...")
model.fit(X_train, y_train)

# 10. Evaluate on unseen test data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nModel training completed!")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")

print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    target_names=["Stayed", "Churned"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 11. Save the trained model for our web app
joblib.dump(model, "churn_model.joblib")

print("\nModel saved as churn_model.joblib")
