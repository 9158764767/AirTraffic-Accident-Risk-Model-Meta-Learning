"""
calculate_shap_v2.py
Calculates SHAP values for the Iteration 2 MAML ONNX model.
Generates an Influence/Summary plot for the dashboard.
"""

import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(BASE_DIR, 'accident_prediction_v2.onnx')
X_TEST_PATH = os.path.join(BASE_DIR, 'X_test_v2.npy')
METRICS_PATH = os.path.join(BASE_DIR, 'model_metrics.json')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

def run_shap():
    print("🚀 Starting SHAP Explainability Analysis for Iteration 2...")

    # 1. Load data and feature names
    if not os.path.exists(X_TEST_PATH) or not os.path.exists(METRICS_PATH):
        print("ERROR: X_test_v2.npy or model_metrics.json missing. Run train_v2.py first.")
        return

    X_test = np.load(X_TEST_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    feature_names = metrics.get('features', [f'feature_{i}' for i in range(X_test.shape[1])])

    # 2. Setup ONNX runtime predict function
    print(f"  Loading ONNX model: {ONNX_PATH}")
    sess = ort.InferenceSession(ONNX_PATH)
    input_name = sess.get_inputs()[0].name

    def predict_fn(X):
        # Ensure X is float32 and has correct shape
        preds = []
        for i in range(len(X)):
            inp = {input_name: X[i:i+1].astype(np.float32)}
            out = sess.run(None, inp)[0]
            preds.append(out.flatten()[0])
        return np.array(preds)

    # 3. Initialize SHAP Explainer
    # Since it's a black-box ONNX model, we use KernelExplainer
    # We'll use a small background set and a subset of test data for speed
    print("  Initializing KernelExplainer (this may take a few minutes)...")
    background = X_test[:20]  # Small baseline
    explainer = shap.KernelExplainer(predict_fn, background)

    # 4. Calculate SHAP values for a subset of the test data
    test_subset = X_test[20:70]  # 50 samples for explanation
    shap_values = explainer.shap_values(test_subset)

    # 5. Generate and Save Plot 1: Summary Bar Plot (Global Importance)
    print("  Generating SHAP Bar Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, test_subset, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("MAML Engine: Feature Importance (SHAP)")
    plt.tight_layout()
    bar_path = os.path.join(ASSETS_DIR, 'v2_shap_bar.png')
    plt.savefig(bar_path, dpi=120)
    plt.close()
    print(f"  ✅ Saved: {bar_path}")

    # 6. Generate and Save Plot 2: Summary Dot Plot (Influence/Direction)
    print("  Generating SHAP Summary (Dot) Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, test_subset, feature_names=feature_names, show=False)
    plt.title("MAML Engine: Feature Influence Sensitivity")
    plt.tight_layout()
    dot_path = os.path.join(ASSETS_DIR, 'v2_shap_summary.png')
    plt.savefig(dot_path, dpi=120)
    plt.close()
    print(f"  ✅ Saved: {dot_path}")

    # 8. NEW: Generate and Save Plot 3: Waterfall Plot (Individual Decision Journey)
    # We'll explain the first sample in the subset
    print("  Generating SHAP Waterfall Plot (Sample #1)...")
    plt.figure(figsize=(10, 6))
    # We need to wrap the values in an Explanation object for the waterfall plot
    exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=test_subset[0],
        feature_names=feature_names
    )
    shap.plots.waterfall(exp, show=False)
    plt.title("Sample Decision Journey (Waterfall)")
    plt.tight_layout()
    waterfall_path = os.path.join(ASSETS_DIR, 'v2_shap_waterfall.png')
    plt.savefig(waterfall_path, dpi=120)
    plt.close()
    print(f"  ✅ Saved: {waterfall_path}")

    # 9. NEW: Generate and Save Plot 4: Heatmap Plot (Comprehensive Overview)
    print("  Generating SHAP Heatmap Plot...")
    plt.figure(figsize=(12, 8))
    # Wrap all subset values for the heatmap
    exp_all = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=test_subset,
        feature_names=feature_names
    )
    shap.plots.heatmap(exp_all, show=False)
    plt.title("Model Decision Heatmap (Total Influence)")
    plt.tight_layout()
    heatmap_path = os.path.join(ASSETS_DIR, 'v2_shap_heatmap.png')
    plt.savefig(heatmap_path, dpi=120)
    plt.close()
    print(f"  ✅ Saved: {heatmap_path}")

    # 7. Export summary stats to JSON for the dashboard
    importance = np.abs(shap_values).mean(0)
    shap_dict = {
        name: float(imp) for name, imp in zip(feature_names, importance)
    }
    # Sort by importance
    shap_dict = dict(sorted(shap_dict.items(), key=lambda item: item[1], reverse=True))

    with open(os.path.join(ASSETS_DIR, 'shap_data.json'), 'w') as f:
        json.dump(shap_dict, f, indent=4)
    print("  ✅ Saved: shap_data.json")

    print("\n✨ SHAP Analysis Complete!")

if __name__ == '__main__':
    run_shap()
