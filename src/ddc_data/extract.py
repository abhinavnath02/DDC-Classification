"""Replay extraction from the delivered raw API snapshots; no network required."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'sources'

def text(value):
    if isinstance(value,str):return re.sub(r'\s+',' ',value).strip()
    if isinstance(value,list):return ' '.join(filter(None,map(text,value)))
    if isinstance(value,dict):return text(value.get('value',''))
    return ''

def seq(value):return value if isinstance(value,list) else []

def main():
    books=defaultdict(list);works={}
    for line in (SRC/'api_snapshots.jsonl').read_text().splitlines():
        snapshot=json.loads(line)
        payload=snapshot.get('response',snapshot.get('payload',{}))
        for doc in payload.get('docs',[]):
            key=str(doc.get('key','')).removeprefix('/works/')
            books[key].append((doc,snapshot))
        if payload.get('type',{}).get('key')=='/type/work':
            works[payload.get('key','').split('/')[-1]]=payload
    editions=json.loads((SRC/'edition_evidence.json').read_text())
    prior=json.loads((SRC/'prior_splits.json').read_text())
    with (SRC/'sample_review.csv').open() as f:
        reviewed={r['document_id']:r['reviewer_status'] for r in csv.DictReader(f)}
    result=[]
    for key,pairs in sorted(books.items()):
        docs=[d for d,_ in pairs]
        codes=sorted({str(c) for d in docs for c in seq(d.get('ddc'))})
        subjects=sorted({text(s) for d in docs for s in seq(d.get('subject')) if text(s)})
        genres=[s for s in subjects if re.search(r'\b(fiction|poetry|poems|drama|romance|fantasy|thriller|mystery|horror|biography|autobiography|memoir|comics|graphic novels|short stories)\b',s,re.I)]
        edition_ids=sorted({e for d in docs for e in seq(d.get('edition_key'))})
        first=max((text(d.get('first_sentence')) for d in docs),key=len,default='')
        first_sources=[snap.get('url',snap.get('source','')) for doc,snap in pairs if text(doc.get('first_sentence'))==first and first]
        work=works.get(key,{})
        if not first and text(work.get('first_sentence')):
            first=text(work['first_sentence']);first_sources=[f'https://openlibrary.org/works/{key}.json']
        description=text(work.get('description'))
        description_url=f'https://openlibrary.org/works/{key}.json' if description else ''
        for edition_id in edition_ids:
            ed=editions.get(edition_id,{}).get('record',{})
            linked='/works/'+key in [x.get('key') for x in ed.get('works',[]) if isinstance(x,dict)]
            english='/languages/eng' in [x.get('key') for x in ed.get('languages',[]) if isinstance(x,dict)]
            if not linked or not english:continue
            if not first and text(ed.get('first_sentence')):
                first=text(ed['first_sentence']);first_sources=[f'https://openlibrary.org/books/{edition_id}.json']
            candidate=text(ed.get('description'))
            if len(candidate)>len(description):
                description=candidate;description_url=f'https://openlibrary.org/books/{edition_id}.json'
        result.append({'document_id':key,'title':max((text(d.get('title')) for d in docs),key=len,default=''),
            'subtitle':max((text(d.get('subtitle')) for d in docs),key=len,default=''),
            'ddc_raw':json.dumps(codes),'subject_headings':json.dumps(subjects,ensure_ascii=False),
            'genre_tags':json.dumps(genres,ensure_ascii=False),'first_sentence':first,
            'first_sentence_sources':json.dumps(sorted(set(first_sources))),
            'description_optional':description,'description_source_url':description_url,
            'edition_ids':json.dumps(edition_ids),'source_url':f'https://openlibrary.org/works/{key}',
            'retrieved_at':max((s.get('retrieved_at','') for _,s in pairs)),
            'review_status':reviewed.get(key,'not_reviewed'),
            'existing_split':prior.get(key,'not_previously_split')})
    with (SRC/'extracted_metadata.jsonl').open('w') as f:
        for row in result:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(f'Extracted {len(result)} unique source work IDs from raw API snapshots.')

if __name__=='__main__':main()
