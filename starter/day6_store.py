"""Dense retrieval on the Day 6 pgvector store (OpenAI query embedding, exact cosine).

Smoke check from repo root:  python starter/day6_store.py "How do I leave an infinite loop?"
Errors (DB down, missing key, model/dimension mismatch) propagate; no mock fallback.
"""
import os
import sys
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / '.env')
FILTER_COLUMNS = {'page', 'chapter'}  # same whitelist idea as Day 6; values are bound parameters
PRICE_PER_M_TOKENS = 0.02  # USD, text-embedding-3-small list price


class DenseRetriever:
    def __init__(self, strategy: str | None = None) -> None:
        self.url = os.environ['DATABASE_URL']
        self.model = os.environ['EMBEDDING_MODEL']
        self.strategy = strategy or os.environ.get('DAY6_STRATEGY', 'B')
        self.client = OpenAI()
        self.embed_tokens = 0
        with psycopg.connect(self.url, connect_timeout=5) as conn:
            dim = conn.execute("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                               "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'").fetchone()[0]
            models = [r[0] for r in conn.execute("SELECT DISTINCT model FROM chunks WHERE strategy = %s",
                                                 (self.strategy,))]
        self.dimensions = int(dim.strip('vector()'))
        if models != [self.model]:
            raise RuntimeError(f'index strategy {self.strategy} built with {models}, query model is {self.model}')

    def embed(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=[text])
        self.embed_tokens += resp.usage.total_tokens
        vec = resp.data[0].embedding
        if len(vec) != self.dimensions:
            raise RuntimeError(f'query vector has {len(vec)} dims, index has {self.dimensions}')
        return vec

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        where, params = ['strategy = %s'], [self.strategy]
        for key, value in (filters or {}).items():
            if key not in FILTER_COLUMNS:
                raise ValueError(f'filter on {key!r} not allowed')
            where.append(f'{key} = %s')
            params.append(value)
        q = '[' + ','.join(map(str, self.embed(query))) + ']'
        with psycopg.connect(self.url, connect_timeout=5) as conn:
            rows = conn.execute(
                "SELECT chunk_id, document_id, source, page, printed_page, chapter, sections, text, "
                f"1 - (embedding <=> %s::vector) FROM chunks WHERE {' AND '.join(where)} "
                "ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s", [q, *params, q, top_k]).fetchall()
        return [{'id': r[0], 'source_id': r[1], 'source': r[2], 'page': r[3], 'text': r[7], 'score': float(r[8]),
                 'rank': i + 1, 'strategy': 'dense',
                 'metadata': {'printed_page': r[4], 'chapter': r[5], 'sections': r[6]}} for i, r in enumerate(rows)]


if __name__ == '__main__':
    r = DenseRetriever()
    print(f'model={r.model} dims={r.dimensions} strategy={r.strategy}')
    for c in r.search(sys.argv[1] if len(sys.argv) > 1 else 'How do I leave an infinite loop?', top_k=3):
        print(f"{c['rank']}. {c['score']:.3f} {c['id']} p{c['page']} | {' '.join(c['text'].split())[:90]}")
    print(f'embedding tokens={r.embed_tokens} cost=${r.embed_tokens / 1e6 * PRICE_PER_M_TOKENS:.6f}')
