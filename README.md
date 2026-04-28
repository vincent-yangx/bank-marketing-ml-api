# Bank Marketing Subscription Prediction API

## Project Overview

This project builds a simple machine learning inference service for predicting whether a bank client will subscribe to a term deposit.

The project is designed as a small end-to-end MLE practice project. It includes:

- A baseline machine learning model using scikit-learn
- Data preprocessing with `ColumnTransformer`
- Model saving and loading with `joblib`
- A FastAPI inference API
- Local API testing through Swagger UI and `curl`

The goal of the first two days is not to build the best possible model, but to create a clean and runnable machine learning project structure.

---

## Dataset

This project uses the UCI Bank Marketing dataset.

The task is binary classification:

- `yes`: the client subscribed to a term deposit
- `no`: the client did not subscribe to a term deposit

The input features used by the model are:

```python
[
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day_of_week",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome"
]
```

---

## Project Structure

```text
bank_marketing_ml_api/
├── data/
├── models/
│   └── bank_marketing_baseline.pkl
├── src/
│   ├── data_loader.py
│   ├── train.py
│   ├── infer.py
│   ├── app.py
│   ├── evaluate.py
│   └── utils.py
├── sample_request.json
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Environment Setup

Create and activate a conda environment:

```bash
conda create -n mle-project python=3.10
conda activate mle-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include:

```txt
pandas
numpy
scikit-learn
joblib
ucimlrepo
fastapi
uvicorn
pydantic
```

---

# Day 1: Build a Scikit-Learn Baseline

## Goal

The goal of Day 1 is to build a clean baseline machine learning pipeline.

By the end of Day 1, the project should be able to:

1. Load the dataset
2. Split the data into train and test sets
3. Preprocess categorical and numerical features
4. Train a baseline model
5. Evaluate the model
6. Save the trained pipeline

---

## Model

The baseline model uses:

- `OneHotEncoder` for categorical features
- `StandardScaler` for numerical features
- `LogisticRegression` as the classifier
- `class_weight="balanced"` to handle class imbalance

The full model is saved as a scikit-learn `Pipeline`, which includes both preprocessing and classification.

This is important because the inference API can later load the same pipeline and make predictions directly from raw JSON input.

---

## Training

Run the training script from the project root directory:

```bash
python src/train.py
```

Do not run this command from inside the `src/` folder.

---

## Expected Training Output

After running the training script, the terminal should print information such as:

```text
Dataset shape
Target distribution
Categorical columns
Numerical columns

Evaluation Metrics:
Accuracy
Precision
Recall
F1 Score
ROC-AUC

Confusion Matrix
Classification Report
```

The script should also save the trained model to:

```text
models/bank_marketing_baseline.pkl
```

---

## Day 1 Completion Checklist

Day 1 is complete if the following items are done:

- [ ] Conda environment created
- [ ] Project folder created
- [ ] Project structure created
- [ ] Dependencies installed
- [ ] `src/data_loader.py` implemented
- [ ] `src/train.py` implemented
- [ ] Model training script runs successfully
- [ ] Evaluation metrics are printed
- [ ] Trained model is saved to `models/bank_marketing_baseline.pkl`
- [ ] Git commit completed

Example Git commit:

```bash
git add .
git commit -m "Add sklearn baseline for bank marketing classification"
```

---

# Day 2: Build a FastAPI Inference API

## Goal

The goal of Day 2 is to turn the trained model into a local inference API.

The inference flow is:

```text
JSON input
    ↓
FastAPI /predict endpoint
    ↓
Load trained sklearn pipeline
    ↓
Convert input into pandas DataFrame
    ↓
Run model.predict() and model.predict_proba()
    ↓
Return prediction and probability
```

---

## Main Files

Day 2 mainly adds two files:

```text
src/infer.py
src/app.py
```

---

## `src/infer.py`

The purpose of `infer.py` is to load the saved model and run prediction logic.

Example structure:

```python
import joblib
import pandas as pd


MODEL_PATH = "models/bank_marketing_baseline.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    return model


def predict_one(model, input_data: dict):
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0]
        class_labels = model.classes_

        prob_dict = {
            str(label): float(prob)
            for label, prob in zip(class_labels, probability)
        }
    else:
        prob_dict = None

    return {
        "prediction": str(prediction),
        "probability": prob_dict,
    }
```

### Explanation of `class_labels`

`model.classes_` stores the class labels learned by the model during training.

For this project, it is usually:

```python
["no", "yes"]
```

The output of `predict_proba()` follows the same order as `model.classes_`.

For example:

```python
model.classes_ = ["no", "yes"]
model.predict_proba(input_df)[0] = [0.82, 0.18]
```

This means:

```json
{
  "no": 0.82,
  "yes": 0.18
}
```

Therefore, `zip(class_labels, probability)` is used to correctly match each probability with its corresponding label.

---

## `src/app.py`

The purpose of `app.py` is to expose the model through FastAPI.

Recommended version:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.infer import load_model, predict_one


app = FastAPI(
    title="Bank Marketing Prediction API",
    description="A FastAPI service for predicting whether a client will subscribe to a term deposit.",
    version="1.0.0",
)


class BankMarketingRequest(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day_of_week: str
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str


model = load_model()


@app.get("/")
def root():
    return {
        "message": "Bank Marketing Prediction API is running."
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(request: BankMarketingRequest):
    try:
        input_data = request.model_dump()
        result = predict_one(model, input_data)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )
```

---

## Run the API

From the project root directory, run:

```bash
uvicorn src.app:app --reload
```

If the server starts correctly, the terminal should show:

```text
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

Keep this terminal open while testing the API.

---

## Open API Documentation

Open the following URL in your browser:

```text
http://127.0.0.1:8000/docs
```

This opens the automatically generated Swagger UI.

You can test the API through:

```text
POST /predict
```

---

## Example Request

Create a file named `sample_request.json` in the project root directory:

```json
{
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1200,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day_of_week": 15,
    "month": "may",
    "duration": 180,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
  }
```

---

## Example Response

A successful response may look like this:

```json
{
  "prediction": "no",
  "probability": {
    "no": 0.82,
    "yes": 0.18
  }
}
```

The exact probability values may be different depending on the trained model.

---

## Test with PowerShell

On Windows PowerShell, use `curl.exe` instead of `curl`:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "@sample_request.json"
```

Important notes:

- Use `http://`, not `heep://`
- Use `curl.exe`, not `curl`
- Keep the FastAPI server running in another terminal
- Run the command from the project root directory where `sample_request.json` exists

---

## Common Errors

### 1. `columns are missing`

Example:

```json
{
  "detail": "Prediction failed: columns are missing: {'day_of_week'}"
}
```

This means the input JSON does not contain all columns used during training.

The prediction input must include exactly the feature columns expected by the trained pipeline.

For this project, the required fields are:

```python
[
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day_of_week",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome"
]
```

---

### 2. `Could not connect to server`

Example:

```text
curl: (7) Failed to connect to 127.0.0.1 port 8000
```

This means the FastAPI server is not running.

Start the server first:

```bash
uvicorn src.app:app --reload
```

Then open another terminal to send the request.

---

### 3. PowerShell `@sample_request.json` error

If PowerShell reports an error related to `@sample_request.json`, use:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "@sample_request.json"
```

Do not use plain `curl` in PowerShell.

---

## Day 2 Completion Checklist

Day 2 is complete if the following items are done:

- [ ] `src/infer.py` created
- [ ] `src/app.py` created
- [ ] `requirements.txt` updated with FastAPI dependencies
- [ ] FastAPI server starts successfully
- [ ] `http://127.0.0.1:8000/docs` opens correctly
- [ ] `/health` endpoint returns `status: ok`
- [ ] `/predict` endpoint returns prediction and probability
- [ ] `sample_request.json` created
- [ ] README updated
- [ ] Git commit completed

Example Git commit:

```bash
git add .
git commit -m "Add FastAPI inference endpoint"
```

---

## Current Project Status

After Day 1 and Day 2, this project has a working local machine learning inference pipeline:

```text
Raw input JSON
    ↓
FastAPI endpoint
    ↓
Saved sklearn pipeline
    ↓
Prediction and probability output
```

## PyTorch MLP Baseline

I also implemented a PyTorch MLP baseline for the same binary classification task.

The PyTorch pipeline includes:

- Dataset and DataLoader
- MLP model
- BCEWithLogitsLoss
- Adam optimizer
- Training loop
- Evaluation loop
- Model checkpoint saving

### PyTorch Results

- Accuracy: 0.9109
- Precision: 0.6556
- Recall: 0.5019
- F1 Score: 0.5685
- ROC-AUC: 0.9308

The model achieved a strong ROC-AUC, suggesting good ranking ability. However, the default threshold of 0.5 gives moderate recall, so future work can tune the classification threshold based on business goals.

### Threshold Tuning

The default threshold of 0.5 achieved an F1 score of 0.5685. After evaluating multiple decision thresholds, threshold = 0.30 achieved the best F1 score of 0.6135, with precision = 0.5429 and recall = 0.7051.

This suggests that lowering the threshold improves recall and overall F1, which is useful in a marketing scenario where identifying more potential subscribers can be valuable.
