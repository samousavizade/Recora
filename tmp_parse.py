import json
import pathlib
path = 'examples/time_series_recora_category.ipynb'
with open(path, encoding='utf-8') as f:
    data = json.load(f)
start, end = 20, 60
with open('notebook_excerpt.txt', 'w', encoding='utf-8') as out:
    for idx, cell in enumerate(data['cells']):
        if idx < start or idx > end:
            continue
        if cell.get('cell_type') != 'code':
            continue
        text = ''.join(cell.get('source', [])).strip()
        if not text:
            continue
        out.write(f"\nCell {idx}\n")
        out.write('-' * 40 + '\\n')
        for line in text.split('\\n')[:40]:
            safe = line.encode('utf-8', 'replace').decode('utf-8')
            out.write(safe + '\\n')
