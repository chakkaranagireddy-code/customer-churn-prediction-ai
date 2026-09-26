import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Sample customer dataset
data = {
    "age": [25, 34, 45, 52, 23, 40, 36, 29, 50, 31,
            27, 48, 33, 41, 26, 55, 38, 30, 46, 35],

    "tenure": [2, 12, 36, 48, 1, 24, 18, 5, 60, 10,
               3, 42, 15, 30, 4, 55, 20, 6, 38, 14],

    "monthly_charges": [80, 60, 45, 50, 95, 55, 65, 90, 40, 75,
                        88, 48, 70, 52, 100, 35, 62, 85, 47, 68],

    "contract": [
        "Month-to-month", "One year", "Two year", "Two year",
        "Month-to-month", "One year", "One year", "Month-to-month",
        "Two year", "Month-to-month", "Month-to-month", "Two year",
        "One year", "Two year", "Month-to-month", "Two year",
        "One year", "Month-to-month", "Two year", "One year"
    ],

    "internet_service": [
        "Fiber", "DSL", "DSL", "Fiber", "Fiber",
        "DSL", "Fiber", "Fiber", "DSL", "Fiber",
        "Fiber", "DSL", "Fiber", "DSL", "Fiber",
        "DSL", "Fiber", "Fiber", "DSL", "Fiber"
    ],

    "churn": [
        1, 0, 0, 0, 1,
        0, 1, 1, 0, 1,
        1, 0, 0, 0, 1,
        0, 0, 1, 0, 1
    ]
}


# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df.drop("churn", axis=1)
y = df["churn"]

# Feature types
categorical_features = ["contract", "internet_service"]
numeric_features = ["age", "tenure", "monthly_charges"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Machine Learning pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Customer Churn Prediction AI")
print("--------------------------------")
print("Model Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "churn_model.joblib")

print("\nModel saved successfully as churn_model.joblib")
