import json
from pathlib import Path

path = Path("examples/time_series_recora_category.ipynb")
data = json.loads(path.read_text(encoding="utf-8"))

with open("cell17_excerpt.txt", "w", encoding="utf-8") as out:
    for idx in range(16, 19):
        cell = data["cells"][idx]
        if cell.get("cell_type") != "code":
            continue
        out.write(f"\nCell {idx}\n")
        out.write("-" * 40 + "\n")
        for line in cell.get("source", []):
            out.write(line)
