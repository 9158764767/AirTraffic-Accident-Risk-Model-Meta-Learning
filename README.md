# AirTraffic-Accident-Risk-Model-Meta-Learning
Machine learning model for aviation accident risk prediction using ACAS and air traffic complexity data. Includes ONNX model export for deployment.
<p align="center">
  <img src="https://img.shields.io/badge/Aviation-Safety-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Machine-Learning-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ONNX-Deployment-green?style=for-the-badge" />
</p>
<p align="center">
  <img src="asset/banner.png" width="100%">
</p>
<h1 align="center">✈️ ONNX-Based Air Crash Probability Analysis</h1>
<h3 align="center">A Predictive Aviation Safety System using EASA ATC Public Data</h3>

---

## 📌 Abstract

This project presents a machine learning-based aviation safety system designed to estimate air crash probability using Air Traffic Control (ATC) public datasets from EASA.

The system analyzes flight path fluctuations, airspeed variations, and geographic movement patterns to identify high-risk conditions. The trained predictive models are exported using **ONNX (Open Neural Network Exchange)** to enable platform-independent deployment and real-time inference capability.

This work demonstrates an end-to-end pipeline from raw aviation data ingestion to deployment-ready predictive modeling.

---

## 🏷 Badges

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![ONNX](https://img.shields.io/badge/ONNX-Deployment-green)
![License](https://img.shields.io/badge/License-Educational-lightgrey)
![Last Commit](https://img.shields.io/github/last-commit/YOUR_USERNAME/ADAI-Accident-Prediction)

> Replace `YOUR_USERNAME` with your GitHub username.

---

## 🎯 Research Objectives

- Analyze aviation flight path fluctuations
- Engineer predictive safety features
- Train accident probability classification models
- Export models to ONNX for cross-platform compatibility
- Identify geographic high-risk zones
- Explore automated “Mayday” alert triggers

---

## 📊 Dataset Description

**Source:** EASA Air Traffic Control Public Data  

Features include:
- Latitude
- Longitude
- Airspeed
- Flight path deviation metrics
- Traffic complexity indicators

The dataset aggregates information from multiple ATC sources.

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Missing value handling
- Feature normalization
- Removal of irrelevant variables
- Merging ACAS + Traffic Complexity datasets

### 2️⃣ Exploratory Data Analysis
- Distribution analysis
- Geographic trend visualization
- Correlation analysis
- Outlier detection

### 3️⃣ Feature Engineering
- Path instability metrics
- Speed fluctuation indicators
- Spatial risk mapping

### 4️⃣ Model Training
- Supervised classification approach
- Train-test split
- Model validation

### 5️⃣ ONNX Conversion
- Export trained model to ONNX
- Enable inference outside Python ecosystem

---

## 📈 Model Evaluation

To assess predictive performance, the following metrics are evaluated:

### 🔹 Accuracy
Overall correctness of the model.

### 🔹 Precision
Measures how many predicted high-risk cases were actually high-risk.

### 🔹 Recall (Sensitivity)
Measures how many actual high-risk cases were correctly identified.

### 🔹 F1-Score
Harmonic mean of Precision and Recall.

### 🔹 ROC-AUC Score
Evaluates the model's ability to distinguish between safe and high-risk flights across different classification thresholds.

---

## 📊 Confusion Matrix

The confusion matrix provides detailed classification breakdown:

|                | Predicted Safe | Predicted Risk |
|----------------|---------------|---------------|
| Actual Safe    | True Negative  | False Positive |
| Actual Risk    | False Negative | True Positive |

- **False Positives:** False alarm situations  
- **False Negatives:** Dangerous undetected cases (critical in aviation safety)

In safety-critical systems, minimizing **False Negatives** is especially important.

---

## 📈 ROC Curve Analysis

The ROC curve plots:

- True Positive Rate (Recall)
- False Positive Rate

A model with strong predictive power will show:
- Curve close to top-left corner
- AUC score close to 1.0

This ensures reliable discrimination between safe and high-risk flight conditions.

---

## 🚀 How to Run

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
```

### 2️⃣ Activate
Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### 3️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 4️⃣ Execute Notebooks
- `ATC.ipynb` → EDA & preprocessing
- `onnx_1.ipynb` → Model training
- `ONNX_2.ipynb` → Advanced risk analysis

---

## 📦 Deployment Capability

The use of **ONNX** enables:

- Cross-language inference (C++, Java, etc.)
- Cloud deployment
- Edge deployment
- Real-time aviation monitoring systems

---

## 🔮 Future Work

- Real-time streaming integration using Apache Kafka
- Distributed processing using Apache Spark Streaming
- Live risk monitoring dashboards
- Automated alert system for aviation authorities
- Integration into ATC decision support systems

---

## 🏗 Project Architecture
## 🏗 System Architecture

<p align="center">
  <img src="asset/architecture.png" width="80%">
</p>
```
Raw ATC Data
      ↓
Data Cleaning
      ↓
EDA & Feature Engineering
      ↓
ML Classification Model
      ↓
ONNX Conversion
      ↓
Risk Prediction & Alert Simulation
```

---

## 📁 Project Structure

```
ADAI_Proj/
│
├── ATC.ipynb
├── onnx_1.ipynb
├── ONNX_2.ipynb
├── data/
├── models/
├── requirements.txt
└── README.md
```

---

## 🧪 Research Contribution

This project demonstrates:

- Applied AI in safety-critical systems
- End-to-end ML engineering pipeline
- Aviation risk modeling
- Model portability and deployment design
- Practical ONNX implementation

---

## 👤 Author

Abhishek Hirve
Artificial Intelligence & Machine Learning  
Focused on aviation safety and applied AI systems.

---

## 📜 License

This project is intended for educational and research purposes.

