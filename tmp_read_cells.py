import json
with open("notebooks/08_reporting/01_technical_report.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)
cells = nb["cells"]
for i in range(54, 92):
    c = cells[i]
    src = "".join(c["source"])
    print(f"\n{'='*60}")
    print(f"CELL [{i:02d}] TYPE={c['cell_type']}")
    print("SOURCE:")
    print(src[:3000])
