import os
import joblib
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
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

from src.data_loader import load_bank_marketing_data

class BankMarketingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)
        
def main():

    X, y = load_bank_marketing_data()
    y = y.values.ravel()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    categorical_cols = X.select_dtypes(include = ["object"]).columns
    numerical_cols = X.select_dtypes(exclude = ["object"]).columns

    preprocessor = ColumnTransformer(
        transformers = [
            ("cat", OneHotEncoder(handle_unknown = "ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )   

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.2, random_state = 42, stratify = y_encoded)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    train_dataset = BankMarketingDataset(X_train_processed, y_train)
    test_dataset = BankMarketingDataset(X_test_processed, y_test)

    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 64)

    input_dim = X_train_processed.shape[1]
    model = MLPClassifier(input_dim)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    num_epoches = 40
    for epoch in range (num_epoches):
        model.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epoches}, Loss: {avg_loss:.4f}")
    
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            logits = model(batch_X)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()

            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())

    print("\nThreshold Tuning:")

    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
        threshold_preds = (np.array(all_probs) >= threshold).astype(int)

        precision = precision_score(all_labels, threshold_preds, zero_division=0)
        recall = recall_score(all_labels, threshold_preds)
        f1 = f1_score(all_labels, threshold_preds)

        print(
            f"Threshold: {threshold:.2f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)

    print("\nEvaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")

    os.makedirs("models", exist_ok = True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "label_encoder_classes": label_encoder.classes_
        },
        "models/torch_mlp.pth"
    )

    joblib.dump(preprocessor, "models/torch_preprocessor.pkl")
    print("\nModel and preprocessor saved to models/torch_mlp.pth and models/torch_preprocessor.pkl")

if __name__ == "__main__":
    main()
