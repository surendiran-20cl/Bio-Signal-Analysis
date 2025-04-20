import gradio as gr
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("bagging_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names in the order used during training
features = [
    'gender_M', 'Gtp', 'triglyceride', 'height(cm)', 'hemoglobin',
    'weight(kg)', 'age', 'waist(cm)', 'HDL', 'fasting blood sugar',
    'ALT', 'systolic', 'LDL', 'relaxation', 'Cholesterol'
]

# Prediction function
def predict_smoking(*args):
    input_df = pd.DataFrame([args], columns=features)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    return "Smoker" if prediction == 1 else "Non-Smoker"

# Create UI
inputs = [gr.Number(label=feature) for feature in features]

gr.Interface(
    fn=predict_smoking,
    inputs=inputs,
    outputs="text",
    title="Bio-Signal Smoking Classifier",
    description="Predict smoking status based on biological health indicators."
).launch()
