"""
Iteration 2 — Dashboard Data Injector
======================================
Reads iteration2/model_metrics.json + lists iteration2/assets/ plots
and SAFELY injects <script id="iter2-data"> into dashboard/index.html
without touching the existing dashboard-data or geo-data tags.
"""

import os, json, re

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR   = os.path.dirname(BASE_DIR)
DASH_HTML  = os.path.join(PROJ_DIR, 'dashboard', 'index.html')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

def build_iter2_data():
    # Load metrics
    metrics_path = os.path.join(BASE_DIR, 'model_metrics.json')
    if not os.path.exists(metrics_path):
        print("ERROR: iteration2/model_metrics.json not found. Run train_v2.py first.")
        return None

    with open(metrics_path) as f:
        m = json.load(f)

    # List plots
    plot_files = sorted([
        f for f in os.listdir(ASSETS_DIR)
        if f.endswith('.png') and f.startswith('v2_')
    ]) if os.path.exists(ASSETS_DIR) else []

    # Build gallery paths (relative to dashboard/)
    gallery = [f'../iteration2/assets/{f}' for f in plot_files]
    print(f"  Found {len(gallery)} plots: {plot_files}")

    data = {
        "version": "Iteration 2",
        "precision": m.get('precision', 0),
        "recall":    m.get('recall', 0),
        "f1_score":  m.get('f1_score', 0),
        "roc_auc":   m.get('roc_auc', 0),
        "threshold": m.get('threshold', 0.5),
        "test_samples": m.get('test_samples', 0),
        "iter1_metrics": m.get('iter1_comparison', {}),
        "improvements": m.get('improvements', []),
        "history": m.get('history', {}),
        "pr_curve": m.get('pr_curve', {}),
        "roc_curve": m.get('roc_curve', {}),
        "confusion_matrix": m.get('confusion_matrix', [[0,0],[0,0]]),
        "plot_gallery": gallery
    }
    return data


def inject_into_html(data):
    with open(DASH_HTML, encoding='utf-8') as f:
        content = f.read()

    new_tag = f'<script id="iter2-data" type="application/json">{json.dumps(data)}</script>'

    # If iter2-data tag already exists, replace it
    existing = re.search(
        r'<script[^>]*id="iter2-data"[^>]*>.*?</script>',
        content, re.DOTALL
    )
    if existing:
        content = content[:existing.start()] + new_tag + content[existing.end():]
        print("  Replaced existing iter2-data tag.")
    else:
        # Insert just before </body>
        content = content.replace('</body>', f'{new_tag}\n</body>', 1)
        print("  Injected new iter2-data tag before </body>.")

    with open(DASH_HTML, 'w', encoding='utf-8') as f:
        f.write(content)

    # Verify
    with open(DASH_HTML, encoding='utf-8') as f:
        c2 = f.read()
    print(f"  iter2-data present: {'iter2-data' in c2}")
    print(f"  </body> present: {'</body>' in c2}")
    print(f"  app.js present: {'app.js' in c2}")
    print(f"  File size: {len(c2):,} bytes")


def main():
    print("📦 Building Iteration 2 Dashboard Data...\n")
    data = build_iter2_data()
    if data is None:
        return

    print(f"\n  Metrics: Precision={data['precision']:.3f} "
          f"Recall={data['recall']:.3f} "
          f"F1={data['f1_score']:.3f} "
          f"AUC={data['roc_auc']:.3f}")

    inject_into_html(data)
    print("\n✅ Done! dashboard/index.html updated with iter2-data tag.")
    print("   Refresh the dashboard and click 'Iteration 2' in the sidebar.")


if __name__ == '__main__':
    main()
