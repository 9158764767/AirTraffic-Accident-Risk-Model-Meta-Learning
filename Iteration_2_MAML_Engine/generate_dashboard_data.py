"""
generate_dashboard_data.py
Generates all plots and JSON data for the dashboard based on the
Best Implementation (RA-trained) model.
Outputs:
  - dashboard/plots/training_loss.png
  - dashboard/plots/prediction_distribution.png
  - dashboard/plots/correlation_heatmap.png
  - dashboard/plots/feature_importance.png
  - dashboard/dashboard_data.json  (embedded in index.html)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json, os

# ── Paths ──────────────────────────────────────────────────────────────────
PLOTS_DIR = 'dashboard/plots/best_model'
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use('dark_background')
PALETTE = {'bg': '#0f172a', 'card': '#1e293b', 'accent': '#38bdf8',
           'green': '#22c55e', 'red': '#ef4444', 'text': '#f8fafc',
           'muted': '#94a3b8'}
FIG_KW = dict(facecolor=PALETTE['bg'])

# ── Load & Preprocess ───────────────────────────────────────────────────────
print("Loading data...")
df_trf  = pd.read_csv('data prep/meta_learningdata/final_dataframe_trfcomplexity.csv')
df_acas = pd.read_csv('data prep/meta_learningdata/final_acas_data.csv')

df_acas['is_risk'] = df_acas['RA'].apply(lambda x: 0 if 'Clear of Conflict' in str(x) else 1)
df_acas['Latitude']       = pd.to_numeric(df_acas['Latitude'],       errors='coerce')
df_acas['Longitude']      = pd.to_numeric(df_acas['Longitude'],      errors='coerce')
df_acas['Altitude']       = pd.to_numeric(df_acas['Altitude'],       errors='coerce')
df_acas['Vertical Speed'] = pd.to_numeric(df_acas['Vertical Speed'], errors='coerce')
df_acas.fillna({'Latitude': 35, 'Longitude': -95, 'Altitude': 10000, 'Vertical Speed': 0}, inplace=True)

FEATURES = ['Latitude', 'Longitude', 'Altitude', 'Vertical Speed']
X = df_acas[FEATURES].values
y = df_acas['is_risk'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Train (record epoch losses) ────────────────────────────────────────────
print("Training model...")
model = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16,  1), nn.Sigmoid())

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

X_t = torch.FloatTensor(X_train_s)
y_t = torch.FloatTensor(y_train).view(-1, 1)

losses = []
for epoch in range(100):
    optimizer.zero_grad()
    out  = model(X_t)
    loss = criterion(out, y_t)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 20 == 0:
        print(f"  Epoch {epoch+1}/100  loss={loss.item():.4f}")

# ── Evaluate ───────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    probs = model(torch.FloatTensor(X_test_s)).numpy().flatten()
preds = (probs > 0.5).astype(int)

precision = precision_score(y_test, preds)
recall    = recall_score(y_test, preds)
f1        = f1_score(y_test, preds)
cm        = confusion_matrix(y_test, preds)
print(f"\nPrecision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 – Training Loss Curve
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4), **FIG_KW)
ax.set_facecolor(PALETTE['card'])
ax.plot(range(1, 101), losses, color=PALETTE['accent'], linewidth=2)
ax.fill_between(range(1, 101), losses, alpha=0.15, color=PALETTE['accent'])
ax.set_title('Training Loss per Epoch (Best Implementation)', color=PALETTE['text'], fontsize=13)
ax.set_xlabel('Epoch', color=PALETTE['muted'])
ax.set_ylabel('BCE Loss', color=PALETTE['muted'])
ax.tick_params(colors=PALETTE['muted'])
for sp in ax.spines.values(): sp.set_edgecolor(PALETTE['muted'])
plt.tight_layout()
loss_path = f'{PLOTS_DIR}/training_loss.png'
plt.savefig(loss_path, dpi=120, facecolor=PALETTE['bg'])
plt.close()
print(f"Saved: {loss_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 – Prediction Probability Distribution
# ─────────────────────────────────────────────────────────────────────────────
safe_probs = probs[y_test == 0]
risk_probs = probs[y_test == 1]

fig, ax = plt.subplots(figsize=(9, 4), **FIG_KW)
ax.set_facecolor(PALETTE['card'])
bins = np.linspace(0, 1, 40)
ax.hist(safe_probs, bins=bins, color='#3b82f6', alpha=0.7, label='Safe (No RA)')
ax.hist(risk_probs, bins=bins, color=PALETTE['red'],  alpha=0.7, label='Risk (RA Active)')
ax.axvline(0.5, color='white', linestyle='--', linewidth=1.2, label='Threshold 0.5')
ax.set_title('Prediction Probability Distribution (Best Implementation)', color=PALETTE['text'], fontsize=13)
ax.set_xlabel('Predicted Probability', color=PALETTE['muted'])
ax.set_ylabel('Count', color=PALETTE['muted'])
ax.tick_params(colors=PALETTE['muted'])
ax.legend(framealpha=0.2)
for sp in ax.spines.values(): sp.set_edgecolor(PALETTE['muted'])
plt.tight_layout()
dist_path = f'{PLOTS_DIR}/prediction_distribution.png'
plt.savefig(dist_path, dpi=120, facecolor=PALETTE['bg'])
plt.close()
print(f"Saved: {dist_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 – Correlation Heatmap (features vs. RA risk)
# ─────────────────────────────────────────────────────────────────────────────
df_feat = df_acas[FEATURES + ['is_risk']].copy()
corr = df_feat.corr()

fig, ax = plt.subplots(figsize=(6, 5), **FIG_KW)
ax.set_facecolor(PALETTE['bg'])
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues',
            linewidths=0.5, linecolor=PALETTE['bg'],
            ax=ax, annot_kws={'size': 10, 'color': PALETTE['text']})
ax.set_title('Feature Correlation Heatmap', color=PALETTE['text'], fontsize=13)
ax.tick_params(colors=PALETTE['muted'])
plt.tight_layout()
corr_path = f'{PLOTS_DIR}/correlation_heatmap.png'
plt.savefig(corr_path, dpi=120, facecolor=PALETTE['bg'])
plt.close()
print(f"Saved: {corr_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 – Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4), **FIG_KW)
ax.set_facecolor(PALETTE['card'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Safe', 'Risk'], yticklabels=['Safe', 'Risk'],
            annot_kws={'size': 14, 'color': PALETTE['text']})
ax.set_title('Confusion Matrix', color=PALETTE['text'], fontsize=13)
ax.set_xlabel('Predicted', color=PALETTE['muted'])
ax.set_ylabel('Actual', color=PALETTE['muted'])
ax.tick_params(colors=PALETTE['muted'])
plt.tight_layout()
cm_path = f'{PLOTS_DIR}/confusion_matrix.png'
plt.savefig(cm_path, dpi=120, facecolor=PALETTE['bg'])
plt.close()
print(f"Saved: {cm_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 – Feature Importance (weight magnitude from first layer)
# ─────────────────────────────────────────────────────────────────────────────
with torch.no_grad():
    weights = model[0].weight.numpy()          # shape (32, 4)
importance = np.abs(weights).mean(axis=0)      # mean absolute weight per feature

fig, ax = plt.subplots(figsize=(7, 4), **FIG_KW)
ax.set_facecolor(PALETTE['card'])
colors = [PALETTE['accent'] if imp == importance.max() else PALETTE['muted'] for imp in importance]
bars = ax.bar(FEATURES, importance, color=colors, edgecolor='none')
ax.set_title('Feature Importance (Mean |Weight| in First Layer)', color=PALETTE['text'], fontsize=13)
ax.set_ylabel('Importance', color=PALETTE['muted'])
ax.tick_params(colors=PALETTE['muted'])
for sp in ax.spines.values(): sp.set_edgecolor(PALETTE['muted'])
for bar, imp in zip(bars, importance):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{imp:.3f}', ha='center', va='bottom', color=PALETTE['text'], fontsize=9)
plt.tight_layout()
fi_path = f'{PLOTS_DIR}/feature_importance.png'
plt.savefig(fi_path, dpi=120, facecolor=PALETTE['bg'])
plt.close()
print(f"Saved: {fi_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PRODUCE JSON DATA BLOCK for index.html
# ─────────────────────────────────────────────────────────────────────────────
print("Generating dashboard JSON data block...")

# Correlations with is_risk
corr_with_risk = {col: round(df_feat[col].corr(df_feat['is_risk']), 4) for col in FEATURES}

# Prediction distribution bins for Chart.js line chart
hist_bins = np.linspace(0, 1, 25)
safe_hist, _ = np.histogram(safe_probs, bins=hist_bins, density=True)
risk_hist, _ = np.histogram(risk_probs, bins=hist_bins, density=True)
bin_labels = [round(float(b), 2) for b in hist_bins[:-1]]

# Training stats
total_n = len(df_acas)
risk_pct = round(float(y.mean() * 100), 1)
avg_alt  = round(float(df_acas['Altitude'].mean()), 1)

data_block = {
    "eda_stats": {
        "total_flights": int(total_n),
        "high_risk_percentage": risk_pct,
        "avg_altitude": avg_alt,
        "correlations": corr_with_risk
    },
    "training_metrics": {
        "epochs": list(range(1, 101)),
        "loss": [round(l, 5) for l in losses]
    },
    "prediction_distribution": {
        "bins": bin_labels,
        "non_high_risk": [round(float(v), 4) for v in safe_hist],
        "high_risk":     [round(float(v), 4) for v in risk_hist]
    },
    "model_metrics": {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1_score":  round(f1,        4),
        "test_samples": int(len(y_test))
    },
    "plot_gallery": {
        "training": [
            f"plots/best_model/training_loss.png",
            f"plots/best_model/feature_importance.png"
        ],
        "inference": [
            f"plots/best_model/prediction_distribution.png",
            f"plots/best_model/confusion_matrix.png",
            f"plots/best_model/correlation_heatmap.png"
        ]
    }
}

with open('dashboard/dashboard_data.json', 'w') as f:
    json.dump(data_block, f, indent=2)
print("Saved: dashboard/dashboard_data.json")

# Also update model_metrics.json
with open('dashboard/model_metrics.json', 'w') as f:
    json.dump({**data_block['model_metrics'],
               "target_source": "Resolution Advisory (RA) Indicators",
               "implementation": "Best Implementation (Real Targets)"}, f, indent=4)
print("Updated: dashboard/model_metrics.json")

print("\n✅ All plots and data generated successfully!")
print(f"   Losses start → end: {losses[0]:.4f} → {losses[-1]:.4f}")
print(f"   Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
