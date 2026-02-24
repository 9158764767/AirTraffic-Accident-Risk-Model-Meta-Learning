# Iteration 2: MAML Meta-Learning Engine for ATC Risk

This directory contains the implementation of the Iteration 2 "Meta-Learning Engine" for the Air Traffic Control (ATC) Risk prediction project. 

## 🧠 Overview

Unlike Iteration 1, which used a standard supervised neural network, Iteration 2 implements **Model-Agnostic Meta-Learning (MAML)**. This enables the model to adapt to specific geographic and kinematic flight patterns (e.g., high-risk landing zones vs. rapid ascent scenarios) with very few examples.

### Key Enhancements
- **MAML Algorithm**: Uses a nested optimization loop (Inner & Outer loops) for few-shot adaptation.
- **Deep Architecture**: A 3-hidden-layer MLP (`MAMLNet`) with 128, 64, and 32 units.
- **Layer Normalization**: Stabilizes second-order gradient calculations required for meta-learning.
- **Platt Scaling**: Calibrates the raw output probabilities for reliable risk estimation on the dashboard.
- **Label Smoothing**: Prevents over-confidence and improves generalization to new flight zones.

## 📁 Directory Structure

- `train_v2.py`: The main training script implementing the MAML logic, episodic sampling, and Platt calibration.
- `build_dashboard_data_v2.py`: Generates the performance metrics used by the landing page dashboard.
- `ONNX_2.ipynb`: Jupyter notebook detailing the data processing, model training, and performance analysis.
- `accident_prediction_v2.onnx`: The final trained and exported meta-learning model in ONNX format.
- `model_metrics.json`: Performance metrics specific to this iteration.
- `data/`: Contains the processed datasets required for training.

### 🛠 Utility & Dashboard Scripts
- `calculate_metrics.py`: Computes Precision, Recall, and F1 scores using the Iteration 2 meta-learning data.
- `inject_geo_data.py`: Injects geographic spatial data into the dashboard for visualization.
- `set_complexity.py`: Configures traffic complexity parameters and correlations for the UI.
- `extract_geo.py`: Prepares geo-spatial features used in the episodic training pipeline.
- `generate_dashboard_data.py`: A comprehensive script to build plots and JSON blocks for the main dashboard.
- `requirements.txt`: Lists all Python dependencies needed to run Iteration 2.

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install torch pandas numpy scikit-learn onnx onnxruntime
   ```
2. **Train the Model**:
   ```bash
   python train_v2.py
   ```
3. **Generate Dashboard Data**:
   ```bash
   python build_dashboard_data_v2.py
   ```

## 📊 Performance
The meta-learning approach provides a significant boost in adaptability across diverse geographic regions compared to the Iteration 1 baseline.
