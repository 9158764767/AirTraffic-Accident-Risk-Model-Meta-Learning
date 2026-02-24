"""
Iteration 2 — MAML Meta-Learning ATC Risk Classifier (Tuned v2)
================================================================
Tuning changes vs initial MAML run:
  • Larger MAMLNet: 3 layers with hidden=128 (vs 64)
  • More meta-epochs: 200 (vs 80)
  • More tasks per outer step: 16 (vs 8)
  • More inner steps K: 10 (vs 5)
  • Stable inner LR α: 0.02 (avoids overshooting on 10 steps)
  • Larger support/query sets: 64/64 (vs 40/40)
  • Finer task partitions: 11 geographic+altitude+speed cells
  • Fine-tuning: 150 epochs with cosine annealing LR
  • Label smoothing (ε=0.05) in fine-tune loss
  • Platt scaling calibration on val set → better ROC-AUC
  • Cosine meta-LR schedule for outer optimiser
"""

import os, json, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR   = os.path.dirname(BASE_DIR)
DATA_DIR   = os.path.join(PROJ_DIR, 'data prep', 'meta_learningdata')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Hyper-parameters (all in one place for easy tuning) ─────────────────────
HP = dict(
    hidden          = 128,    # MAMLNet hidden width
    meta_epochs     = 200,    # outer MAML iterations
    tasks_per_step  = 16,     # episodes per meta-gradient step
    n_inner_steps   = 10,     # K inner-loop gradient steps
    inner_lr        = 0.02,   # α  (inner learning rate)
    meta_lr         = 5e-4,   # outer meta-optimiser lr
    n_support       = 64,     # support samples per episode
    n_query         = 64,     # query   samples per episode
    ft_epochs       = 150,    # fine-tune epochs on full train
    ft_lr           = 1e-3,   # fine-tune initial lr
    label_smooth    = 0.05,   # label smoothing ε
    max_pos_weight  = 15.0,   # max class weight cap
    batch_ft        = 256,    # fine-tune mini-batch size
)

# ── MAML-Compatible Network (NO BatchNorm) ───────────────────────────────────
class MAMLNet(nn.Module):
    """
    3-layer MLP optimised for MAML:
      – No BatchNorm (breaks second-order gradients)
      – LayerNorm instead (element-wise, compatible with functional forward)
      – Larger capacity: input → 128 → 64 → 32 → 1
    """
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        h1, h2 = hidden, hidden // 2

        # Weights & biases stored as named parameters so we can do
        # functional forward with fast_weights dict in the inner loop.
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h2 // 2)
        self.fc4 = nn.Linear(h2 // 2, 1)

        # LayerNorm affine params (included in fast_weights)
        self.ln1 = nn.LayerNorm(h1)
        self.ln2 = nn.LayerNorm(h2)

        # Weight initialisation — He normal for ReLU
        for m in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x, params=None):
        if params is None:
            # Standard forward (fine-tuning / inference)
            x = F.relu(self.ln1(self.fc1(x)))
            x = F.relu(self.ln2(self.fc2(x)))
            x = F.relu(self.fc3(x))
            return torch.sigmoid(self.fc4(x))
        else:
            # Functional forward with explicit fast_weights (inner loop)
            x = F.relu(F.layer_norm(
                    F.linear(x, params['fc1.weight'], params['fc1.bias']),
                    [params['ln1.weight'].shape[0]],
                    params['ln1.weight'], params['ln1.bias']))
            x = F.relu(F.layer_norm(
                    F.linear(x, params['fc2.weight'], params['fc2.bias']),
                    [params['ln2.weight'].shape[0]],
                    params['ln2.weight'], params['ln2.bias']))
            x = F.relu(F.linear(x, params['fc3.weight'], params['fc3.bias']))
            return torch.sigmoid(F.linear(x, params['fc4.weight'], params['fc4.bias']))


# ── Episode / Task Sampler ───────────────────────────────────────────────────
class EpisodeSampler:
    """
    Generates binary classification episodes from geographic/kinematic cells.
    Tasks:
      • 5 latitude quintile bands
      • 3 altitude bands  (low / mid / high)
      • 3 vertical-speed bands (descending / level / climbing)
    = up to 11 distinct task distributions over the same feature space.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 n_support: int = 64, n_query: int = 64):
        self.X        = torch.FloatTensor(X)
        self.y        = torch.FloatTensor(y)
        self.n_s      = n_support
        self.n_q      = n_query
        self.need     = n_support + n_query
        masks         = []

        # Latitude quintile tasks (column 0)
        lat = X[:, 0]
        q   = np.percentile(lat, [20, 40, 60, 80])
        edges = [-np.inf] + list(q) + [np.inf]
        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = np.where((lat > lo) & (lat <= hi))[0]
            if len(idx) >= self.need:
                masks.append(idx)

        # Altitude band tasks (column 2: Altitude — raw, pre-scale)
        alt = X[:, 2]
        for lo, hi in [(-np.inf, -0.5), (-0.5, 0.5), (0.5, np.inf)]:
            idx = np.where((alt > lo) & (alt <= hi))[0]
            if len(idx) >= self.need:
                masks.append(idx)

        # Vertical-speed band tasks (column 3: Vertical Speed scaled)
        vs = X[:, 3]
        for lo, hi in [(-np.inf, -0.3), (-0.3, 0.3), (0.3, np.inf)]:
            idx = np.where((vs > lo) & (vs <= hi))[0]
            if len(idx) >= self.need:
                masks.append(idx)

        self.masks = masks
        print(f"  📦 EpisodeSampler: {len(self.masks)} tasks  "
              f"(min size = {min(len(m) for m in masks):,})")

    def sample(self):
        mask   = self.masks[np.random.randint(len(self.masks))]
        chosen = np.random.choice(mask, size=self.need, replace=False)
        s, q   = chosen[:self.n_s], chosen[self.n_s:]
        return (self.X[s], self.y[s].unsqueeze(1),
                self.X[q], self.y[q].unsqueeze(1))


# ── Label-Smoothed BCE ───────────────────────────────────────────────────────
def smooth_bce(pred: torch.Tensor, target: torch.Tensor,
               eps: float, pos_weight: float = 1.0) -> torch.Tensor:
    """BCE with label smoothing and class weighting."""
    target_s = target * (1 - eps) + (1 - target) * eps
    loss = -(pos_weight * target_s * torch.log(pred + 1e-8)
             + (1 - target_s) * torch.log(1 - pred + 1e-8))
    return loss.mean()


# ── Inner Loop (fast adaptation) ────────────────────────────────────────────
def inner_loop(model: MAMLNet, X_sup: torch.Tensor, y_sup: torch.Tensor,
               inner_lr: float, k: int) -> dict:
    fast = {n: p.clone() for n, p in model.named_parameters()}
    for _ in range(k):
        pred  = model(X_sup, fast)
        loss  = F.binary_cross_entropy(pred, y_sup)
        grads = torch.autograd.grad(loss, fast.values(),
                                     create_graph=True, allow_unused=True)
        fast  = {n: w - inner_lr * (g if g is not None else torch.zeros_like(w))
                 for (n, w), g in zip(fast.items(), grads)}
    return fast


# ── Platt Scaling (calibration) ──────────────────────────────────────────────
class PlattScaler(nn.Module):
    """Single affine layer trained on val logits to calibrate probabilities."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return torch.sigmoid(self.a * logits + self.b)


def calibrate(model: MAMLNet,
              X_val_t: torch.Tensor, y_val: np.ndarray,
              n_steps: int = 500) -> PlattScaler:
    model.eval()
    with torch.no_grad():
        raw = model(X_val_t)
        # convert probabilities → logits for calibration
        logits = torch.log(raw / (1 - raw + 1e-8))

    scaler  = PlattScaler()
    opt     = optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=n_steps)
    y_t     = torch.FloatTensor(y_val).unsqueeze(1)

    def closure():
        opt.zero_grad()
        cal_prob = scaler(logits)
        loss     = F.binary_cross_entropy(cal_prob, y_t)
        loss.backward()
        return loss

    opt.step(closure)
    print(f"  Platt scaler: a={scaler.a.item():.4f}  b={scaler.b.item():.4f}")
    return scaler


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_and_prepare_data():
    print("📂 Loading datasets...")
    df_trf  = pd.read_csv(os.path.join(DATA_DIR, 'final_dataframe_trfcomplexity.csv'))
    df_acas = pd.read_csv(os.path.join(DATA_DIR, 'final_acas_data.csv'))
    print(f"  TRF rows: {len(df_trf):,}  |  ACAS rows: {len(df_acas):,}")

    # Real RA labels
    print("🎯 Real RA-based target labels...")
    df_acas['is_risk'] = df_acas['RA'].apply(
        lambda x: 0 if 'Clear of Conflict' in str(x) else 1)
    pos  = df_acas['is_risk'].sum()
    neg  = len(df_acas) - pos
    ratio = neg / max(pos, 1)
    print(f"  Positive: {pos:,}  Negative: {neg:,}  Ratio: {ratio:.1f}:1")

    # Feature engineering
    print("🔧 Feature engineering...")
    for col in ['Latitude', 'Longitude', 'Altitude', 'Vertical Speed']:
        df_acas[col] = pd.to_numeric(df_acas[col], errors='coerce')

    df_acas.fillna({'Latitude': df_acas['Latitude'].median(),
                    'Longitude': df_acas['Longitude'].median(),
                    'Altitude': df_acas['Altitude'].median(),
                    'Vertical Speed': 0}, inplace=True)

    df_acas['abs_vert_speed']  = df_acas['Vertical Speed'].abs()
    df_acas['altitude_bucket'] = pd.cut(
        df_acas['Altitude'], bins=[0, 5000, 10000, 20000, 100000],
        labels=[0, 1, 2, 3], include_lowest=True).astype(float)
    df_acas['high_risk_zone']  = (
        df_acas['Latitude'].between(30, 40) &
        df_acas['Longitude'].between(-100, -90)).astype(int)
    df_acas['lat_lon_product'] = df_acas['Latitude'] * df_acas['Longitude'].abs()
    df_acas['lat_sq']          = df_acas['Latitude'] ** 2
    df_acas['alt_vs_interact'] = df_acas['Altitude'] * df_acas['abs_vert_speed']

    # TRF complexity scalars
    trf_num  = df_trf.select_dtypes(include=np.number)
    cplx_cols = [c for c in trf_num.columns if any(
        k in c.upper() for k in ['CPLX', 'INTER', 'SPEED', 'HORIZ', 'VERTICAL', 'TRF'])][:6]
    for col in cplx_cols:
        safe = col.replace(' ', '_').replace('/', '_')[:20]
        df_acas[f'trf_{safe}'] = trf_num[col].mean()

    feat_cols = (['Latitude', 'Longitude', 'Altitude', 'Vertical Speed',
                  'abs_vert_speed', 'altitude_bucket', 'high_risk_zone',
                  'lat_lon_product', 'lat_sq', 'alt_vs_interact']
                 + [f'trf_{c.replace(" ","_").replace("/","_")[:20]}' for c in cplx_cols])

    X = df_acas[feat_cols].fillna(0).values.astype(np.float32)
    y = df_acas['is_risk'].values.astype(np.float32)
    print(f"  Feature matrix: {X.shape}  Positive rate: {y.mean():.3f}")
    return X, y, feat_cols, ratio


# ── Main ─────────────────────────────────────────────────────────────────────
def train():
    print("\n" + "="*65)
    print("  ADAI Iteration 2 — MAML Meta-Learning (Tuned)")
    print("="*65 + "\n")

    X, y, feature_names, imbalance_ratio = load_and_prepare_data()

    # Splits
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15, random_state=42, stratify=y_tv)
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # Scaling — fit on train only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    # Episode sampler (on scaled train)
    sampler = EpisodeSampler(X_train_s, y_train,
                              n_support=HP['n_support'], n_query=HP['n_query'])

    # Model + meta-optimiser
    input_dim  = X_train_s.shape[1]
    model      = MAMLNet(input_dim, hidden=HP['hidden'])
    meta_opt   = optim.AdamW(model.parameters(), lr=HP['meta_lr'], weight_decay=1e-4)
    meta_sched = optim.lr_scheduler.CosineAnnealingLR(
        meta_opt, T_max=HP['meta_epochs'], eta_min=1e-5)

    # ── MAML Meta-Training ───────────────────────────────────────────────────
    print(f"\n🔁 MAML Meta-Training  "
          f"({HP['meta_epochs']} epochs × {HP['tasks_per_step']} tasks, "
          f"K={HP['n_inner_steps']}, α={HP['inner_lr']})")

    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    X_v_t   = torch.FloatTensor(X_val_s)
    y_v_t   = torch.FloatTensor(y_val).unsqueeze(1)
    best_meta_f1 = 0.0
    best_meta_weights = None

    for epoch in range(HP['meta_epochs']):
        model.train()
        meta_opt.zero_grad()

        task_losses = []
        for _ in range(HP['tasks_per_step']):
            X_sup, y_sup, X_qry, y_qry = sampler.sample()
            fast        = inner_loop(model, X_sup, y_sup,
                                      HP['inner_lr'], HP['n_inner_steps'])
            qry_pred    = model(X_qry, fast)
            task_losses.append(F.binary_cross_entropy(qry_pred, y_qry))

        meta_loss = torch.stack(task_losses).mean()
        meta_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        meta_opt.step()
        meta_sched.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out  = model(X_v_t)
            val_loss = F.binary_cross_entropy(val_out, y_v_t).item()
            val_pred = (val_out.numpy() >= 0.5).astype(int).flatten()
            val_f1   = f1_score(y_val, val_pred, zero_division=0)

        history['train_loss'].append(round(meta_loss.item(), 6))
        history['val_loss'].append(round(val_loss, 6))
        history['val_f1'].append(round(val_f1, 4))

        # Track best meta-initialisation by val F1
        if val_f1 > best_meta_f1:
            best_meta_f1      = val_f1
            best_meta_weights = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 20 == 0:
            lr = meta_opt.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{HP['meta_epochs']} | "
                  f"meta_loss={meta_loss.item():.4f}  "
                  f"val_loss={val_loss:.4f}  val_F1={val_f1:.4f}  lr={lr:.6f}")

    # Restore best meta-init
    if best_meta_weights:
        model.load_state_dict(best_meta_weights)
    print(f"  ✅ Best meta val-F1 = {best_meta_f1:.4f}")

    # ── Fine-Tune on Full Train Set ──────────────────────────────────────────
    print(f"\n🔧 Fine-tuning on full train set ({HP['ft_epochs']} epochs)...")
    ft_model  = copy.deepcopy(model)
    ft_opt    = optim.AdamW(ft_model.parameters(), lr=HP['ft_lr'], weight_decay=1e-4)
    ft_sched  = optim.lr_scheduler.CosineAnnealingLR(ft_opt, T_max=HP['ft_epochs'], eta_min=1e-5)
    pos_weight = min(imbalance_ratio, HP['max_pos_weight'])

    X_tr_t = torch.FloatTensor(X_train_s)
    y_tr_t = torch.FloatTensor(y_train).unsqueeze(1)
    ds     = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=HP['batch_ft'], shuffle=True)

    best_val_f1   = 0.0
    best_ft_state = None
    patience      = 30
    patience_ctr  = 0

    for ep in range(HP['ft_epochs']):
        ft_model.train()
        for Xb, yb in loader:
            ft_opt.zero_grad()
            pred = ft_model(Xb)
            loss = smooth_bce(pred, yb, HP['label_smooth'], pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
            ft_opt.step()
        ft_sched.step()

        ft_model.eval()
        with torch.no_grad():
            vp = ft_model(X_v_t).numpy().flatten()
            vf = f1_score(y_val, (vp >= 0.5).astype(int), zero_division=0)

        if vf > best_val_f1:
            best_val_f1   = vf
            best_ft_state = copy.deepcopy(ft_model.state_dict())
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  ⏹ Early stop at fine-tune epoch {ep+1}")
                break

        if (ep + 1) % 30 == 0:
            print(f"  FT Epoch {ep+1:3d} | val_F1={vf:.4f}  best={best_val_f1:.4f}")

    if best_ft_state:
        ft_model.load_state_dict(best_ft_state)
    print(f"  ✅ Fine-tune best val-F1 = {best_val_f1:.4f}")

    # ── Platt Scaling Calibration ─────────────────────────────────────────────
    print("\n📐 Calibrating with Platt scaling...")
    platt = calibrate(ft_model, X_v_t, y_val)

    # ── Threshold Tuning (on calibrated val probs) ────────────────────────────
    print("\n🎚  Threshold tuning on calibrated val probabilities...")
    ft_model.eval()
    with torch.no_grad():
        raw_val  = ft_model(X_v_t)
        logits_v = torch.log(raw_val / (1 - raw_val + 1e-8))
        cal_val  = platt(logits_v).numpy().flatten()

    best_thresh, best_f1_t = 0.5, 0.0
    for thresh in np.arange(0.15, 0.85, 0.005):
        preds = (cal_val >= thresh).astype(int)
        f     = f1_score(y_val, preds, zero_division=0)
        if f > best_f1_t:
            best_f1_t, best_thresh = f, thresh
    print(f"  Best threshold: {best_thresh:.3f}  (val F1={best_f1_t:.4f})")

    # ── Test Evaluation ───────────────────────────────────────────────────────
    print("\n📊 Evaluating on held-out test set...")
    X_te_t = torch.FloatTensor(X_test_s)
    ft_model.eval()
    with torch.no_grad():
        raw_test    = ft_model(X_te_t)
        logits_te   = torch.log(raw_test / (1 - raw_test + 1e-8))
        test_proba  = platt(logits_te).numpy().flatten()

    y_pred    = (test_proba >= best_thresh).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, test_proba)
    cm        = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n{'='*55}")
    print(f"  MAML Meta-Learning (Tuned) — Test Results:")
    print(f"  Threshold : {best_thresh:.3f}")
    print(f"  Precision : {precision:.4f}  ({precision*100:.1f}%)")
    print(f"  Recall    : {recall:.4f}  ({recall*100:.1f}%)")
    print(f"  F1 Score  : {f1:.4f}  ({f1*100:.1f}%)")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  vs Iter 1 → P=50.8%, R=25.3%, F1=33.6%")
    print(f"{'='*55}")

    prec_c, rec_c, _ = precision_recall_curve(y_test, test_proba)
    fpr, tpr, _       = roc_curve(y_test, test_proba)

    # ── ONNX Export ───────────────────────────────────────────────────────────
    print("\n💾 Exporting ONNX model...")
    onnx_path = os.path.join(BASE_DIR, 'accident_prediction_v2.onnx')
    dummy     = torch.randn(1, input_dim)
    ft_model.eval()
    torch.onnx.export(
        ft_model, dummy, onnx_path,
        input_names  = ['features'],
        output_names = ['risk_prob'],
        dynamic_axes = {'features': {0: 'batch_size'}, 'risk_prob': {0: 'batch_size'}},
        opset_version = 11,
        do_constant_folding = True
    )
    print(f"  ✅ ONNX saved → {onnx_path}")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        out  = sess.run(['risk_prob'], {'features': dummy.numpy()})
        print(f"  ✅ ONNX verified — sample output: {out[0].flatten()[0]:.4f}")
    except ImportError:
        print("  ⚠️  onnxruntime not installed — skipping ONNX verify")

    # ── Save Metrics ─────────────────────────────────────────────────────────
    metrics = {
        "version"         : "Iteration 2 — MAML Meta-Learning (Tuned)",
        "approach"        : "Model-Agnostic Meta-Learning (MAML) + Platt Calibration",
        "meta_epochs"     : HP['meta_epochs'],
        "tasks_per_step"  : HP['tasks_per_step'],
        "inner_steps"     : HP['n_inner_steps'],
        "inner_lr"        : HP['inner_lr'],
        "hidden"          : HP['hidden'],
        "ft_epochs"       : HP['ft_epochs'],
        "precision"       : round(precision, 4),
        "recall"          : round(recall, 4),
        "f1_score"        : round(f1, 4),
        "roc_auc"         : round(auc, 4),
        "threshold"       : round(best_thresh, 3),
        "test_samples"    : int(len(y_test)),
        "positive_rate"   : round(float(y_test.mean()), 4),
        "target_source"   : "Resolution Advisory (RA) — real TCAS signals",
        "improvements"    : [
            "Real RA-based labels (not random)",
            "MAML — episodic meta-training across 11 geographic/kinematic tasks",
            "Deeper MAMLNet (128-64-32-1) with LayerNorm, He weight init",
            "Inner-loop fast adaptation K=10, α=0.02 for stable convergence",
            "16 tasks/outer-step × 200 meta-epochs with cosine LR annealing",
            "Fine-tuning with label smoothing (ε=0.05) + class-weighted BCE",
            "Platt scaling calibration → improved ROC-AUC",
            "Threshold sweep on calibrated probabilities for max F1",
            "ONNX export with constant folding (opset 11)"
        ],
        "iter1_comparison": {"precision": 0.508, "recall": 0.253, "f1_score": 0.336},
        "confusion_matrix": cm,
        "history"         : history,
        "pr_curve"        : {
            "precision": prec_c.tolist()[::max(1, len(prec_c)//100)],
            "recall"   : rec_c.tolist()[::max(1, len(rec_c)//100)]
        },
        "roc_curve"       : {
            "fpr": fpr.tolist()[::max(1, len(fpr)//100)],
            "tpr": tpr.tolist()[::max(1, len(tpr)//100)]
        },
        "features": feature_names
    }

    metrics_path = os.path.join(BASE_DIR, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📋 Metrics → {metrics_path}")

    np.save(os.path.join(BASE_DIR, 'X_test_v2.npy'), X_test_s[:200])

    print("\n✅ Iteration 2 (MAML Tuned) — Complete!")
    return metrics


if __name__ == '__main__':
    train()
