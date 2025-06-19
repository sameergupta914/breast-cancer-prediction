
import gradio as gr
from gradio.themes import Soft as SoftTheme
import pickle
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load models (update paths if models are saved separately)
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("dt_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

# Load dataset to get feature names
data = load_breast_cancer()
feature_names = data.feature_names.tolist()

# Prediction function
def predict_all_models(*features):
    input_data = [features]
    results = {}

    results["SVM"] = "Benign" if svm_model.predict(input_data)[0] == 1 else "Malignant"
    results["KNN"] = "Benign" if knn_model.predict(input_data)[0] == 1 else "Malignant"
    results["Random Forest"] = "Benign" if rf_model.predict(input_data)[0] == 1 else "Malignant"
    results["Decision Tree"] = "Benign" if dt_model.predict(input_data)[0] == 1 else "Malignant"

    return results

# Gradio interface
inputs = [gr.Number(label=feature) for feature in feature_names]

interface = gr.Interface(
    fn=predict_all_models,
    inputs=inputs,
    outputs=gr.JSON(label="Prediction Results"),
    title="Breast Cancer Classifier",
    description="Enter 30 medical features to get predictions from SVM, KNN, Random Forest, and Decision Tree.",
    theme=SoftTheme()
)

interface.launch()
