"""
inject_geo_data.py
Reads dashboard/geo_data.json and injects it as a <script id="geo-data"> tag
into dashboard/index.html, right before the closing </div> of the app container.
Also cleans up dashboard-data correlations (removes self-correlation).
"""
import json, re

# 1. Load geo data
with open('dashboard/geo_data.json', encoding='utf-8') as f:
    geo_raw = f.read().strip()

geo_tag = f'    <script id="geo-data" type="application/json">{geo_raw}</script>\n'

# 2. Load index.html
with open('dashboard/index.html', encoding='utf-8') as f:
    lines = f.readlines()

# 3. Check if geo-data already there
has_geo = any('id="geo-data"' in l for l in lines)
has_dd  = any('id="dashboard-data"' in l for l in lines)

print(f'geo-data already present: {has_geo}')
print(f'dashboard-data present: {has_dd}')

if has_geo:
    print('geo-data already exists — skipping injection')
else:
    # Insert geo-data tag just before the dashboard-data script tag (or before </div></body>)
    out = []
    injected = False
    for line in lines:
        if not injected and 'id="dashboard-data"' in line:
            out.append(geo_tag)
            injected = True
        out.append(line)

    if not injected:
        # Fallback: insert before last </div> or </body>
        for i in range(len(out)-1, -1, -1):
            if '</div>' in out[i] or '</body>' in out[i]:
                out.insert(i, geo_tag)
                injected = True
                break

    if injected:
        with open('dashboard/index.html', 'w', encoding='utf-8') as f:
            f.writelines(out)
        print('SUCCESS: geo-data script tag injected into index.html')
    else:
        print('ERROR: could not find insertion point')

# 4. Also fix correlations in dashboard-data: remove the spurious YEAR=1.0 self-correlation
with open('dashboard/index.html', encoding='utf-8') as f:
    content = f.read()

m = re.search(r'(<script[^>]*id="dashboard-data"[^>]*>)(.*?)(</script>)', content, re.DOTALL)
if m:
    prefix, raw, suffix = m.group(1), m.group(2).strip(), m.group(3)
    try:
        d = json.loads(raw)
        corr = d['eda_stats']['correlations']
        # Remove any key with value 1.0 (self-correlation artifact)
        cleaned = {k: v for k, v in corr.items() if abs(v) < 0.9999}
        d['eda_stats']['correlations'] = cleaned
        print(f'Cleaned correlations: {cleaned}')
        new_raw = json.dumps(d)
        new_content = content[:m.start()] + prefix + new_raw + suffix + content[m.end():]
        with open('dashboard/index.html', 'w', encoding='utf-8') as f:
            f.write(new_content)
        print('SUCCESS: correlations cleaned in index.html')
    except Exception as e:
        print(f'Correlation clean failed: {e}')

print('Done!')
