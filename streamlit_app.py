import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

@st.cache_data
def load_data():
    """Loads the breast cancer dataset."""
    data = pd.read_csv('Breast_cancer_data.csv')
    return data

@st.cache_resource
def train_model(X, y):
    """Trains the Decision Tree model."""
    model = DecisionTreeClassifier(max_depth=50, min_samples_leaf=8, min_samples_split=2, random_state=42)
    model.fit(X, y)
    return model

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Breast Cancer Predictor", page_icon="♋", layout="wide")

    st.title("Breast Cancer Prediction")
    st.write("Enter the patient's medical measurements to predict whether the diagnosis is malignant or benign. This app uses a Decision Tree Classifier.")

    # Load and prepare data
    data = load_data()
    X = data[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']]
    y = data['diagnosis']

    # Train the model
    model = train_model(X, y)

    # Get user input in the sidebar
    st.sidebar.header("Patient's Measurements")

    def user_input_features():
        mean_radius = st.sidebar.slider("Mean Radius", float(X['mean_radius'].min()), float(X['mean_radius'].max()), float(X['mean_radius'].mean()), 0.01)
        mean_texture = st.sidebar.slider("Mean Texture", float(X['mean_texture'].min()), float(X['mean_texture'].max()), float(X['mean_texture'].mean()), 0.01)
        mean_perimeter = st.sidebar.slider("Mean Perimeter", float(X['mean_perimeter'].min()), float(X['mean_perimeter'].max()), float(X['mean_perimeter'].mean()), 0.01)
        mean_area = st.sidebar.slider("Mean Area", float(X['mean_area'].min()), float(X['mean_area'].max()), float(X['mean_area'].mean()), 0.1)
        mean_smoothness = st.sidebar.slider("Mean Smoothness", float(X['mean_smoothness'].min()), float(X['mean_smoothness'].max()), float(X['mean_smoothness'].mean()), 0.0001)

        data = {'mean_radius': mean_radius,
                'mean_texture': mean_texture,
                'mean_perimeter': mean_perimeter,
                'mean_area': mean_area,
                'mean_smoothness': mean_smoothness}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Display user input
    st.subheader("Patient's Input:")
    st.write(input_df)

    # Predict
    if st.sidebar.button("Predict"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction")
        if prediction[0] == 1:
            st.success("The diagnosis is **Benign**.")
        else:
            st.error("The diagnosis is **Malignant**.")

        st.subheader("Prediction Probability")
        st.write(f"Benign: {prediction_proba[0][1]*100:.2f}%")
        st.write(f"Malignant: {prediction_proba[0][0]*100:.2f}%")

if __name__ == '__main__':
    main()
