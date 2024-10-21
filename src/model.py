import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib  # For loading your saved model

# Load your dataset
normalized_credit = pd.read_csv('../docs/encoded_creditcard_data_normalized.csv')

# Prepare features and target
X_creditcard_normalized = normalized_credit.drop(columns=['Class', 'Unnamed: 0'])
y_creditcard_normalized = normalized_credit['Class']

# Split the dataset into training and testing sets
X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(
    X_creditcard_normalized, 
    y_creditcard_normalized, 
    test_size=0.2, 
    random_state=42
)

# Set the MLflow tracking URI (optional)
mlflow.set_tracking_uri("http://localhost:5000")  # Adjust this if needed

# Start an MLflow run
with mlflow.start_run():
    # Load the saved model
    random_forest_model = joblib.load('../models/random_forest_model.pkl')  # Update the path accordingly

    # Evaluate the model
    y_pred_rf = random_forest_model.predict(X_test_credit)
    accuracy_rf = accuracy_score(y_test_credit, y_pred_rf)
    roc_auc_rf = roc_auc_score(y_test_credit, random_forest_model.predict_proba(X_test_credit)[:, 1])

    # Log parameters and metrics
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_metric("accuracy", accuracy_rf)
    mlflow.log_metric("roc_auc", roc_auc_rf)

    print(f"Random Forest Accuracy: {accuracy_rf}")
    print(f"Random Forest ROC-AUC: {roc_auc_rf}")
