import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score, 
    recall_score,
    f1_score, 
    roc_auc_score,
    confusion_matrix, 
    classification_report
)

from data_loader import load_bank_marketing_data

def main():
    # load data
    X, y = load_bank_marketing_data()
    y = y.values.ravel()

    print("Dataset Shape:", X.shape)
    print("Target distribution:")
    print(y)
    

    categorical_cols = X.select_dtypes(include = ["object"]).columns
    numerical_cols = X.select_dtypes(exclude = ["object"]).columns

    preprocessor = ColumnTransformer(
        transformers = [
            ("cat", OneHotEncoder(handle_unknown = "ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )

    model = Pipeline(
        steps = [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter  = 1000,
                class_weight = "balanced",
                random_state = 42,
            )
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42, stratify = y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="yes")
    recall = recall_score(y_test, y_pred, pos_label="yes")
    f1 = f1_score(y_test, y_pred, pos_label="yes")
    roc_auc = roc_auc_score((y_test == "yes").astype(int), y_prob)

    print("\nEvaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    model_path = "models/bank_marketing_baseline.pkl"
    joblib.dump(model, model_path)

    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()




