import joblib

model = joblib.load("models/bank_marketing_baseline.pkl")
print(list(model.named_steps["preprocessor"].feature_names_in_))