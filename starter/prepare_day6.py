"""Export Day 6 chunks (pgvector) + benchmark to the Day 7 JSON contract.

Run from repo root:  python starter/prepare_day6.py
Writes starter/data/{chunks,questions,cases}.json. Vectors stay in pgvector (dense
retrieval queries the Day 6 table directly); only text/metadata are exported.
"""
import json
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import hit  # same hit definition as the Day 7 baseline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'starter' / 'data'


def export_chunks(url, strategy):
    with psycopg.connect(url, connect_timeout=5) as conn:
        rows = conn.execute(
            "SELECT chunk_id, document_id, source, page, printed_page, chapter, chapter_title, sections, "
            "char_start, char_end, model, text FROM chunks WHERE strategy = %s ORDER BY page, char_start",
            (strategy,)).fetchall()
    return [{'id': r[0], 'source_id': r[1], 'source': r[2], 'page': r[3], 'text': r[11],
             'metadata': {'printed_page': r[4], 'chapter': r[5], 'chapter_title': r[6], 'sections': r[7],
                          'char_start': r[8], 'char_end': r[9], 'day6_strategy': strategy, 'embedding_model': r[10]}}
            for r in rows]


def convert_question(q):
    gold = [{'source_id': q['gold_source_id'], 'page': q['gold_page'], 'evidence': q['evidence_quote']}]
    gold += [{'source_id': q['gold_source_id'], 'page': a['page'], 'evidence': a['quote']} for a in q.get('alt_evidence', [])]
    return {'id': q['id'], 'split': q['split'], 'type': q.get('type'), 'question': q['question'],
            'expected_answer': q['expected_answer'], 'gold': gold, 'filter': q.get('filter')}


def main():
    day6 = Path(os.environ.get('DAY6_DIR', ROOT.parent / 'K5-L3A-Day06-NguyenNgocToan-2A202603405-DataIngestion')) / 'starter'
    load_dotenv(ROOT / 'starter' / '.env')
    strategy = os.environ.get('DAY6_STRATEGY', 'B')
    chunks = export_chunks(os.environ['DATABASE_URL'], strategy)
    if not chunks or len({c['id'] for c in chunks}) != len(chunks):
        raise SystemExit(f'strategy {strategy}: empty or duplicate chunk IDs')
    raw = [json.loads(l) for l in open(day6 / 'benchmark' / 'questions.jsonl', encoding='utf-8') if l.strip()]
    questions = [convert_question(q) for q in raw]
    unanswerable = json.loads((ROOT / 'data' / 'unanswerable.json').read_text(encoding='utf-8'))

    # Report gold alternatives that do not sit whole inside one chunk (evidence split by chunking).
    problems = []
    for q in questions:
        ok = [g for g in q['gold'] if any(hit(c, [g]) for c in chunks)]
        if len(ok) != len(q['gold']):
            problems.append((q['id'], len(ok), len(q['gold'])))
    # Unanswerable questions must not share IDs with answerable ones.
    assert not {q['id'] for q in questions} & {u['id'] for u in unanswerable}

    cases = [{'id': q['id'], 'split': q['split'], 'question': q['question'], 'answerable': True,
              'expected_behavior': 'search_and_answer', 'filter': q['filter']} for q in questions]
    cases += [{**u, 'filter': None} for u in unanswerable]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, obj in (('chunks.json', chunks), ('questions.json', questions), ('cases.json', cases)):
        (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for split in ('dev', 'test'):
        a = sum(c['split'] == split and c['answerable'] for c in cases)
        u = sum(c['split'] == split and not c['answerable'] for c in cases)
        print(f'{split}: {a} answerable + {u} unanswerable')
    print(f'chunks: {len(chunks)} (strategy {strategy}); gold not inside one chunk: {problems or "none"}')


if __name__ == '__main__':
    main()
