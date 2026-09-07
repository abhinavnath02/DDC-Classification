"""Rebuild clean data and audit from the frozen extraction snapshot, using Python 3.9+.

Run from repository root: python3 -m src.ddc_data.etl
This script does not train a model or contact a network service.
"""
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'sources'
DATA=ROOT/'data'
AUDIT=ROOT/'audit'
LABELS={f'{i}00':n for i,n in enumerate([
 'Computing, information and general works','Philosophy and psychology','Religion',
 'Social sciences','Language','Science','Technology and applied subjects',
 'Arts and recreation','Literature','History and geography'])}

def clean(value):
    text=unicodedata.normalize('NFC',str(value or ''))
    return re.sub(r'\s+',' ',''.join(' ' if unicodedata.category(c)=='Cc' else c for c in text)).strip()

def norm(value):return re.sub(r'\W+',' ',clean(value).casefold()).strip()

def array(value):
    if isinstance(value,list):return value
    try:
        x=json.loads(value)
        return x if isinstance(x,list) else []
    except (ValueError,TypeError):return []

def base(value):
    if isinstance(value,str) and re.fullmatch(r'[0-9]{3}(?:[./][0-9]+)*',value.strip()):
        return value.strip()[0]+'00'
    return None

def write_csv(path,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def model_text(row):
    # No identifiers, DDC labels, collection queries, review notes or authors enter the input.
    parts=[row['title']]
    if row['subtitle'] and norm(row['subtitle']) not in norm(row['title']):parts.append(row['subtitle'])
    if row['first_sentence']:parts.append(row['first_sentence'])
    if row['subject_headings']:parts.append('Subjects: '+'; '.join(row['subject_headings']))
    return '\n'.join(parts)

def main():
    DATA.mkdir(exist_ok=True);AUDIT.mkdir(exist_ok=True)
    source=[json.loads(line) for line in (SRC/'extracted_metadata.jsonl').read_text().splitlines()]
    decisions=json.loads((SRC/'decisions.json').read_text())
    evidence=json.loads((SRC/'edition_evidence.json').read_text())
    authors=defaultdict(set)
    for line in (SRC/'api_snapshots.jsonl').read_text().splitlines():
        snap=json.loads(line);payload=snap.get('response',snap.get('payload',{}))
        for doc in payload.get('docs',[]):
            key=doc.get('key','').split('/')[-1]
            authors[key].update(clean(a) for a in doc.get('author_name',[]) if clean(a))
    eligible={};quarantine=[];enrichment_audit=[]
    for original in source:
        key=original['document_id'];codes=array(original['ddc_raw']);bases={base(x) for x in codes}
        reasons=[]
        if not re.fullmatch(r'OL[0-9]+W',key):reasons.append('invalid_work_id')
        if not clean(original['title']):reasons.append('missing_title')
        if not codes or None in bases:reasons.append('missing_or_unrecognized_DDC')
        if len(bases-{None})!=1:reasons.append('conflicting_or_missing_main_class')
        if key in decisions['quarantine']:reasons.append(decisions['quarantine'][key])
        if reasons:
            quarantine.append({**original,'release_exclusion_reason':'; '.join(reasons)});continue
        label=next(iter(bases));override=decisions['main_class_corrections'].get(key)
        if override:label=override['label']
        row={'document_id':key,'source_work_ids':[key],'title':clean(original['title']),
             'subtitle':clean(original['subtitle']),'authors':sorted(authors[key]),
             'subject_headings':list(dict.fromkeys(clean(x) for x in array(original['subject_headings']) if clean(x))),
             'genre_tags':list(dict.fromkeys(clean(x) for x in array(original['genre_tags']) if clean(x))),
             'first_sentence':clean(original['first_sentence']),
             'first_sentence_sources':array(original.get('first_sentence_sources','[]')),
             'ddc_raw':codes,'label_level_1':label,'label_name':LABELS[label],
             'edition_ids':array(original['edition_ids']),'isbns':[],
             'source_urls':[original['source_url']], 'retrieved_at':original['retrieved_at'],
             'review_status':original['review_status'],'label_override':override or {},
             'previous_split_memberships':{key:original['existing_split']},
             'description_optional':clean(original['description_optional']),
             'description_source_url':original['description_source_url'],
             'text_enrichment_sources':[]}
        matches=[]
        for edition_id in row['edition_ids']:
            entry=evidence.get(edition_id,{});ed=entry.get('record',{})
            if '/works/'+key not in [w.get('key') for w in ed.get('works',[]) if isinstance(w,dict)]:continue
            row['isbns']+=ed.get('isbn_13',[])+ed.get('isbn_10',[])
            language=[v.get('key') for v in ed.get('languages',[]) if isinstance(v,dict)]
            # Unknown or foreign-language editions do not supply new text.
            if '/languages/eng' in language and norm(row['title']) in norm(ed.get('title','')):
                matches.append((edition_id,ed))
        if matches:
            edition_id,ed=max(matches,key=lambda pair:len(clean(pair[1].get('title','')))+len(clean(pair[1].get('subtitle',''))))
            before=(row['title'],row['subtitle'])
            row['title']=clean(ed.get('title')) or row['title']
            row['subtitle']=clean(ed.get('subtitle')) or row['subtitle']
            if before!=(row['title'],row['subtitle']):
                url=f'https://openlibrary.org/books/{edition_id}.json'
                row['text_enrichment_sources'].append(url)
                enrichment_audit.append({'document_id':key,'old_title':before[0],'old_subtitle':before[1],
                                         'new_title':row['title'],'new_subtitle':row['subtitle'],'source':url})
        row['isbns']=sorted(set(row['isbns']))
        eligible[key]=row
    aliases=[]
    for family in decisions['edition_families']:
        ids=[key for key in family if key in eligible]
        if len(ids)<2:continue
        assert len({eligible[k]['label_level_1'] for k in ids})==1
        chosen=max(ids,key=lambda k:(len(model_text(eligible[k])),k))
        canonical=eligible[chosen]
        for key in ids:
            aliases.append({'source_work_id':key,'canonical_document_id':chosen,'action':'consolidated_edition_family',
                            'reason':'Reviewed matching author/title/edition family; one training example, all source IDs retained.'})
            if key==chosen:continue
            other=eligible.pop(key)
            for field in ['source_work_ids','authors','subject_headings','genre_tags','ddc_raw','edition_ids','isbns','source_urls','text_enrichment_sources']:
                canonical[field]=sorted(set(canonical[field]+other[field]))
            canonical['previous_split_memberships'].update(other['previous_split_memberships'])
        canonical['deduplication_status']='consolidated_edition_family'
    records=sorted(eligible.values(),key=lambda x:x['document_id'])
    # Leakage groups conservatively link matching titles, identical inputs or shared editions/ISBNs.
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        parent[max(a,b)]=min(a,b)
    seen={};id_index={}
    for i,row in enumerate(records):
        row['raw_text']=model_text(row)
        row['text_availability']='title_and_context' if row['subtitle'] or row['subject_headings'] or row['first_sentence'] else 'title_only'
        row['deduplication_status']=row.get('deduplication_status','retained_distinct_work')
        keys=[('title',norm(row['title'])),('input',norm(row['raw_text']))]
        keys += [('edition',e) for e in row['edition_ids']]+[('isbn',v) for v in row['isbns']]
        for key in keys:
            if key in seen:union(i,seen[key])
            else:seen[key]=i
        for key in row['source_work_ids']:id_index[key]=i
    for group in decisions['related_keep_separate']:
        members=[id_index[x] for x in group if x in id_index]
        for i in members[1:]:union(members[0],i)
    for i,row in enumerate(records):row['split_group_id']='group_'+records[find(i)]['document_id']
    group_old_splits=defaultdict(set)
    for row in records:
        group_old_splits[row['split_group_id']].update(row['previous_split_memberships'].values())
    for row in records:
        old=group_old_splits[row['split_group_id']]
        row['prior_holdout_status']='previous_test_in_group' if 'test' in old else 'previous_validation_in_group' if 'validation' in old else 'none'
    # Resolve the entire visible title queue; no pending title decisions enter the handoff.
    with (SRC/'title_candidates.csv').open() as f:title_candidates=list(csv.DictReader(f))
    mapping={key:r for r in records for key in r['source_work_ids']}
    title_audit=[]
    for candidate in title_candidates:
        key=candidate['document_id'];row=mapping.get(key)
        eds=[evidence[e]['record'] for e in array(next(x for x in source if x['document_id']==key)['edition_ids']) if e in evidence]
        title_audit.append({'source_work_id':key,'original_title':candidate['title'],
            'authors':sorted(authors[key]),'edition_subtitles':sorted({clean(e.get('subtitle')) for e in eds if clean(e.get('subtitle'))}),
            'edition_urls':[f"https://openlibrary.org{e['key']}.json" for e in eds if e.get('key')],
            'canonical_document_id':row['document_id'] if row else '',
            'decision':row['deduplication_status'] if row else 'quarantined_label_issue',
            'explanation': ('Matching author and edition family consolidated for training.' if row and len(row['source_work_ids'])>1 else
                           'Different authors, subtitles or volumes retained; conservative split grouping prevents identical inputs crossing splits.' if row else decisions['quarantine'].get(key,'Excluded by label validation.'))})
    write_csv(DATA/'books.csv',records)
    with (DATA/'books.jsonl').open('w') as f:
        for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_csv(AUDIT/'quarantine.csv',quarantine)
    write_csv(AUDIT/'duplicate_decisions.csv',title_audit)
    write_csv(AUDIT/'edition_aliases.csv',aliases)
    write_csv(AUDIT/'text_enrichment.csv',enrichment_audit,['document_id','old_title','old_subtitle','new_title','new_subtitle','source'])
    coverage=[{'label_level_1':label,'class_name':name,'books':sum(r['label_level_1']==label for r in records)} for label,name in LABELS.items()]
    write_csv(AUDIT/'class_coverage.csv',coverage)
    counts={'extracted_unique_work_ids':len(source),'release_books':len(records),'quarantined_work_ids':len(quarantine),
            'work_ids_consolidated':sum(len(r['source_work_ids'])-1 for r in records),'reviewed_title_rows':len(title_audit),
            'edition_families_consolidated':len({a['canonical_document_id'] for a in aliases}),
            'L1_corrections':len(decisions['main_class_corrections']),
            'with_subjects':sum(bool(r['subject_headings']) for r in records),
            'with_first_sentence':sum(bool(r['first_sentence']) for r in records),
            'with_subtitle':sum(bool(r['subtitle']) for r in records),
            'title_only':sum(r['text_availability']=='title_only' for r in records),
            'text_fields_enriched_from_editions':len(enrichment_audit),
            'scope':'Ten-class metadata ETL release. Labels are source-derived except documented L1 overrides. No ML training or OCR evaluation.'}
    assert len(source)==len(records)+len(quarantine)+counts['work_ids_consolidated']
    (AUDIT/'quality_report.json').write_text(json.dumps(counts,indent=2)+'\n')
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [DATA,AUDIT,SRC] for p in sorted(folder.glob('*')) if p.is_file() and p.name != "manifest.json"}
    (ROOT/'audit/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
