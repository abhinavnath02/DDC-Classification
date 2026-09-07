"""Validate the package contract without training a model."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from load_data import load_data

ROOT=Path(__file__).resolve().parent

def main():
    for relative,digest in json.loads((ROOT/'manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    loaded=load_data();rows=loaded['records']
    assert rows and len(rows)==len({r['document_id'] for r in rows})
    assert set(loaded['y'])=={str(i)+'00' for i in range(10)}
    aliases=[key for row in rows for key in row['source_work_ids']]
    assert len(aliases)==len(set(aliases))
    with (ROOT/'audit/quarantine.csv').open() as f:quarantined=list(csv.DictReader(f))
    assert set(aliases).isdisjoint(r['document_id'] for r in quarantined)
    seen=defaultdict(set)
    for row in rows:
        assert row['raw_text'].strip() and row['title'].strip()
        assert set(row['genre_tags']) <= set(row['subject_headings'])
        assert row['source_urls'] and row['retrieved_at']
        seen[row['raw_text'].casefold()].add(row['split_group_id'])
    assert all(len(groups)==1 for groups in seen.values())
    by_alias={key:r for r in rows for key in r['source_work_ids']}
    assert by_alias['OL17618370W']['document_id']==by_alias['OL44556550W']['document_id']
    assert by_alias['OL19714233W']['document_id']!=by_alias['OL42419347W']['document_id']
    assert by_alias['OL45859692W']['document_id']!=by_alias['OL45859854W']['document_id']
    assert by_alias['OL13386549W']['label_level_1']=='800'
    assert 'OL42539495W' not in by_alias
    with (ROOT/'data/books.csv').open() as f:csv_rows=list(csv.DictReader(f))
    assert [r['label_level_1'] for r in csv_rows]==loaded['y']
    report=json.loads((ROOT/'audit/quality_report.json').read_text())
    assert len(rows)+len(quarantined)+report['work_ids_consolidated']==report['extracted_unique_work_ids']
    print(f'PASS: {len(rows)} examples; checksums, labels, aliases, exclusions, text groups, CSV/JSON agreement and reviewed edge cases.')

if __name__=='__main__':main()
