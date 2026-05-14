import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.svm import SVC

from preprocess import (
    load_data,
    preprocess_data
)


# Create required folders
os.makedirs("model", exist_ok=True)

os.makedirs("data/processed", exist_ok=True)


# Load Dataset
DATA_PATH = "data/raw/placement_data.csv"

print("Loading dataset...")

df = load_data(DATA_PATH)


# Preprocess Dataset
print("Preprocessing dataset...")

X_train, X_test, y_train, y_test = (
    preprocess_data(df)
)


# Final Model
final_model = SVC(
    C=0.1,
    kernel='linear',
    probability=True,
    random_state=42
)

mlflow.set_experiment(
    "Placement Prediction Experiment"
)

# MLflow Tracking
with mlflow.start_run():

    print("Training model...")

    final_model.fit(
        X_train,
        y_train
    )

    # Predictions
    predictions = final_model.predict(
        X_test
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # ROC-AUC
    probabilities = (
        final_model.predict_proba(X_test)[:, 1]
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    # Classification Report
    report = classification_report(
        y_test,
        predictions
    )

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )

    # Print Results
    print("\nAccuracy:", accuracy)

    print("\nROC-AUC Score:", roc_auc)

    print("\nClassification Report:\n")

    print(report)

    print("\nConfusion Matrix:\n")

    print(cm)

    # MLflow Logging
    mlflow.log_param(
        "model",
        "SVC"
    )

    mlflow.log_param(
        "C",
        0.1
    )

    mlflow.log_param(
        "kernel",
        "linear"
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    mlflow.sklearn.log_model(
    final_model,
    "mlflow_model"
    )

    # Save Model
    joblib.dump(
        final_model,
        "model/model.pkl"
    )

    print("\nModel saved successfully.")

    print("MLflow tracking completed.")


print("\nTraining pipeline completed successfully.")