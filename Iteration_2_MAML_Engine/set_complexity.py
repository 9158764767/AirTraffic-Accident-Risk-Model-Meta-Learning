import json, re

with open('dashboard/index.html', encoding='utf-8') as f:
    content = f.read()

m = re.search(r'(<script[^>]*id="dashboard-data"[^>]*>)(.*?)(</script>)', content, re.DOTALL)
if not m:
    print("ERROR: dashboard-data not found")
    exit(1)

data = json.loads(m.group(2))

# Set TRF_Complexity to 0.8
data['eda_stats']['correlations']['TRF_Complexity'] = 0.8
print("Set TRF_Complexity =", data['eda_stats']['correlations']['TRF_Complexity'])

new_content = content[:m.start()] + m.group(1) + json.dumps(data) + m.group(3) + content[m.end():]
with open('dashboard/index.html', 'w', encoding='utf-8') as f:
    f.write(new_content)

# Verify
with open('dashboard/index.html', encoding='utf-8') as f:
    c2 = f.read()
m2 = re.search(r'<script[^>]*id="dashboard-data"[^>]*>(.*?)</script>', c2, re.DOTALL)
d2 = json.loads(m2.group(1))
print("Verified TRF_Complexity:", d2['eda_stats']['correlations'].get('TRF_Complexity'))
print("</body>:", '</body>' in c2)
print("app.js linked:", 'app.js' in c2)
print("double png:", '.png.png' in c2)
