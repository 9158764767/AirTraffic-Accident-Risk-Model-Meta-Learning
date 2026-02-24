# Explainable ATC MAML Engine (SHAP)

This package contains **Iteration 2** of the Air Traffic Control (ATC) Risk Prediction model, featuring **MAML (Model-Agnostic Meta-Learning)** and **SHAP (SHapley Additive exPlanations)** for decision transparency.

## 🌟 Key Features
- **MAML Meta-Learning**: A model designed for rapid adaptation to new aviation sectors and traffic conditions.
- **SHAP Explainability**: Deep insights into why the model classifies a flight as "High Risk".
- **ONNX Weights**: Pre-trained model in an interoperable format for high-performance inference.

## 📂 Package Contents
- `train_v2.py`: The core MAML training architecture.
- `calculate_shap_v2.py`: The explainability engine that generates feature importance and sensitivity plots.
- `accident_prediction_v2.onnx`: Optimized pre-trained model weights.
- `X_test_v2.npy`: Representative sample data for SHAP analysis.
- `model_metrics.json`: Model performance metadata and feature labels.
- `assets/`: Folder containing generated SHAP bar and summary plots.
- `requirements.txt`: Python dependencies.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run SHAP Explainability Analysis
To re-generate the SHAP values and visualizations:
```bash
python calculate_shap_v2.py
```
This will update the plots in the `assets/` folder.

## 📊 Understanding the results

### Global Importance (`v2_shap_bar.png`)
Shows the overall ranking of features. For example, if **Altitude** is at the top, it means the model's risk prediction is most sensitive to flight height.

### Feature Sensitivity (`v2_shap_summary.png`)
Shows the *direction* of influence.
- **Red dots** (High value) on the right mean the feature increases risk.
- **Blue dots** (Low value) on the right mean the feature increases risk.

### Decision Journey (`v2_shap_waterfall.png`)
Shows the contribution of each feature to a **single specific prediction**.
- **Positive values (Pink)**: Features that pushed the risk higher for this flight.
- **Negative values (Blue)**: Features that decreased the risk for this flight.
- **E[f(x)]**: The base risk level.
- **f(x)**: The final predicted risk probability.

### Comprehensive Overview (`v2_shap_heatmap.png`)
Provides a **global summary** of all predictions in the sample set.
- Each column represents one flight.
- Darker colors indicate stronger feature influence.
- This is useful for spotting patterns across common high-risk flight profiles.

---
*Developed for the ADAI Project - Air Traffic Control Risk Classification.*
*CC ABHISHEK HIRVE.*
