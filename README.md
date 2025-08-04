# Breast Cancer Prediction App

A Streamlit web application for predicting breast cancer diagnosis (benign vs. malignant) based on user-specified tumor measurements. This project demonstrates end-to-end data ingestion, model training, and deployment of multiple machine-learning classifiers in an interactive dashboard.

---

## 🔗 Live Demo

👉 [View the app online](https://breastcancerprediction-sg.streamlit.app/)  

---

## 📂 Repository

[https://github.com/sameergupta914/breast-cancer-prediction](https://github.com/sameergupta914/breast-cancer-prediction)

---

## 🚀 Features

- **Interactive Sliders**  
  Adjust key tumor metrics (radius, texture, perimeter, area, smoothness, etc.) in real time.
- **Multiple Classifiers**  
  Predict with Decision Tree, Random Forest, K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) models.
- **Comparison Table**  
  View side-by-side model predictions and confidence scores for the same inputs.
- **Instant Feedback**  
  Get immediate diagnostic predictions as soon as you click **Predict**.

---

## 📊 Dataset

- The app uses the [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) (UCI).  
- All features are precomputed summary statistics of digitized images of fine needle aspirate (FNA) of breast masses.

| Feature         | Description                                                |
| --------------- | ---------------------------------------------------------- |
| mean_radius     | Average distance from center to tumor boundary             |
| mean_texture    | Standard deviation of gray-scale values                    |
| mean_perimeter  | Mean size of the tumor perimeter                          |
| mean_area       | Mean area of the tumor                                     |
| mean_smoothness | Mean local variation in radius lengths                     |
| …               | …                                                          |

---

## 🛠️ Tech Stack

- **Python 3.8+**  
- **Streamlit** for the web interface  
- **scikit-learn** for model training and inference  
- **pandas & numpy** for data handling  
- **GitHub & Streamlit Cloud** for source control and deployment  

---

## 🔧 Installation & Local Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/sameergupta914/breast-cancer-prediction.git
   cd breast-cancer-prediction
