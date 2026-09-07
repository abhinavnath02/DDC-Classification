"""The ML team can import load_data() without pandas or any extra dependency."""
import json
from pathlib import Path

def load_data(path=None):
    path=Path(path) if path else Path(__file__).resolve().parents[2]/'data/books.jsonl'
    records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    return {
        'X':[r['raw_text'] for r in records],
        'y':[r['label_level_1'] for r in records],
        'groups':[r['split_group_id'] for r in records],
        'records':records,
    }

if __name__=='__main__':
    data=load_data()
    print(f"Loaded {len(data['X'])} examples with string labels: {sorted(set(data['y']))}")
