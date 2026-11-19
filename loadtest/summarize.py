import pandas as pd
from pathlib import Path

rows = []
for n in (1, 2, 3):
    f = Path("results") / f"containers_{n}_stats.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f)
    r = d[d["Name"] == "/predict"].iloc[0]
    rows.append({"API containers": n, "Requests": int(r["Request Count"]), "Failures": int(r["Failure Count"]),
                 "Avg (ms)": round(r["Average Response Time"]), "Median (ms)": round(r["Median Response Time"]),
                 "p95 (ms)": round(r["95%"]), "Max (ms)": round(r["Max Response Time"]),
                 "Req/s": round(r["Requests/s"], 1)})
table = pd.DataFrame(rows).to_markdown(index=False)
Path("results/summary.md").write_text(table)
print(table)
