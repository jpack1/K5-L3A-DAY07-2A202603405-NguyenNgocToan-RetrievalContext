"""Embed advanced chunks into the Day 6 pgvector table as strategy 'ADV' (needs the Day 6 Docker database).

Run from repo root: python advanced/index.py            # OpenAI text-embedding-3-small -> strategy ADV
                    python advanced/index.py --gemini   # free Gemini gemini-embedding-001 (1536-d) -> strategy ADVG
Same model/table as Day 6 (text-embedding-3-small, vector(1536)). Unchanged chunks are not re-embedded;
ADV rows that are no longer in chunks.json are deleted, so the table always mirrors chunks.json.
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / 'starter' / '.env')
GEMINI = '--gemini' in sys.argv
STRATEGY = 'ADVG' if GEMINI else 'ADV'
GEMINI_MODEL, DIMS = 'gemini-embedding-001', 1536  # 1536 = the Day 6 vector column; Gemini truncates (MRL)


def gemini_embed(texts, task):
    """Free-tier Gemini embeddings. The free tier allows 30,000 tokens/minute, so batches stay under
    ~80,000 characters (~20k tokens) and wait a minute between batches."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options={'timeout': 120000})
    batches, cur = [], []
    for t in texts:
        if cur and sum(map(len, cur)) + len(t) > 80000:
            batches.append(cur)
            cur = []
        cur.append(t)
    batches += [cur] if cur else []
    out = []
    for k, batch in enumerate(batches):
        for attempt in range(5):
            try:
                r = client.models.embed_content(model=GEMINI_MODEL, contents=batch,
                                                config=types.EmbedContentConfig(task_type=task, output_dimensionality=DIMS))
                break
            except Exception as e:  # 429: wait for the minute window, then fail loudly
                if attempt == 4:
                    raise
                print('retry', k, str(e)[:100], flush=True)
                time.sleep(65)
        out += [e.values for e in r.embeddings]
        print(f'embedded batch {k + 1}/{len(batches)} ({len(out)} texts)', flush=True)
        if k + 1 < len(batches):
            time.sleep(62)
    return out


MAX_CHARS = 20000  # far below the 8191-token embedding limit


def main():
    chunks = json.loads((ROOT / 'data' / 'chunks.json').read_text(encoding='utf-8'))
    assert len({c['id'] for c in chunks}) == len(chunks), 'duplicate chunk IDs'
    assert max(len(c['text']) for c in chunks) < MAX_CHARS, 'chunk too long for one embedding'
    model, url = (GEMINI_MODEL if GEMINI else os.environ['EMBEDDING_MODEL']), os.environ['DATABASE_URL']
    h = lambda t: hashlib.sha256(t.encode('utf-8')).hexdigest()
    with psycopg.connect(url, connect_timeout=5) as conn:
        have = dict(conn.execute('SELECT chunk_id, text_hash FROM chunks WHERE strategy = %s AND model = %s',
                                 (STRATEGY, model)).fetchall())
    todo = [c for c in chunks if have.get(c['id']) != h(c['text'])]
    tokens, vectors = 0, []
    if GEMINI:
        vectors = gemini_embed([c['text'] for c in todo], 'RETRIEVAL_DOCUMENT') if todo else []
    else:
        client = OpenAI(timeout=60)
        for i in range(0, len(todo), 50):
            r = client.embeddings.create(model=model, input=[c['text'] for c in todo[i:i + 50]])
            tokens += r.usage.total_tokens
            vectors += [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
    with psycopg.connect(url, connect_timeout=5) as conn, conn.cursor() as cur:
        rows = []
        for c, v in zip(todo, vectors):
            meta = c['metadata']
            m = re.match(r'Chapter (\d+)', meta['chapter'])
            rows.append((STRATEGY, c['id'], c['source_id'], c['source'], c['page'], None, int(m.group(1)),
                         meta['chapter'], [meta['modality']] + meta['figure_labels'] + meta['table_labels'],
                         None, None, c['text'], h(c['text']), model, '[' + ','.join(map(str, v)) + ']'))
        cur.executemany(
            'INSERT INTO chunks (strategy, chunk_id, document_id, source, page, printed_page, chapter, chapter_title, '
            'sections, char_start, char_end, text, text_hash, model, embedding) '
            'VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector) ON CONFLICT (strategy, chunk_id) DO UPDATE SET '
            'text = EXCLUDED.text, text_hash = EXCLUDED.text_hash, embedding = EXCLUDED.embedding, '
            'sections = EXCLUDED.sections, chapter = EXCLUDED.chapter, chapter_title = EXCLUDED.chapter_title, '
            'model = EXCLUDED.model', rows)
        cur.execute('DELETE FROM chunks WHERE strategy = %s AND NOT (chunk_id = ANY(%s))',
                    (STRATEGY, [c['id'] for c in chunks]))
        deleted = cur.rowcount
        n = cur.execute('SELECT count(*) FROM chunks WHERE strategy = %s', (STRATEGY,)).fetchone()[0]
    assert n == len(chunks), f'{n} rows stored, {len(chunks)} chunks built'
    stats = {'strategy': STRATEGY, 'model': model, 'chunks': len(chunks), 'embedded_now': len(todo),
             'deleted_obsolete': deleted, 'embedding_tokens': tokens, 'cost_usd': 0.0 if GEMINI else round(tokens / 1e6 * 0.02, 6)}
    (ROOT / 'data' / f'index_stats_{STRATEGY}.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(stats)


if __name__ == '__main__':
    main()
