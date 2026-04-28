import joblib
import pandas as pd

MODEL_PATH = "models/bank_marketing_baseline.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

def predict_one(model, input_data: dict):
    """
    Make prediction for one input sample.

    Args:
        model: trained sklearn pipeline
        input_data: one sample in dictionary format

    Returns:
        prediction result and probability
    """

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
        "probability": prob_dict
    }

def predict_batch(model, input_data_list: list[dict]):
    input_df = pd.DataFrame(input_data_list)

    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)
    class_labels = model.classes_

    results = []

    for pred, prob in zip(predictions, probabilities):
        prob_dict = {
            str(label): float(p)
            for label, p in zip(class_labels, prob)
        }

        results.append({
            "prediction": str(pred),
            "probability": prob_dict
        })

    return results