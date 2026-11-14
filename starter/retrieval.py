"""Retrieval methods for Day 7 + benchmark runner (retrieval/context only, no LLM).

Methods: tfidf (starter), bm25 (lexical), dense (Day 6 pgvector), hybrid (RRF of dense + bm25).
Each can run with the user-provided `filter` (chapter) from the question.

Run from repo root:
    python starter/retrieval.py --split dev --output starter/results/retrieval-dev
"""
import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

from baseline import TfidfIndex, hit, pack_context, tokenize, validate

PRICE_PER_M_TOKENS = 0.02  # text-embedding-3-small


def keep(chunk, filters):
    """Metadata filter on page/chapter, same columns the dense store allows."""
    return all((chunk['page'] if k == 'page' else chunk['metadata'].get(k)) == v for k, v in (filters or {}).items())


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75, bigrams=False):
        self.chunks, self.k1, self.b, self.bigrams = chunks, k1, b, bigrams
        self.tf = [Counter(self._tokens(c['text'])) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(term for t in self.tf for term in t)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def _tokens(self, text):
        toks = tokenize(text)
        # Bigrams let a pasted data list ("6; 7; 7") or phrase ("skewed left") match as a unit;
        # single digits alone appear on hundreds of pages and carry almost no weight.
        return toks + [a + '_' + b for a, b in zip(toks, toks[1:])] if self.bigrams else toks

    def search(self, query, *, filters=None, top_k=3):
        terms = [t for t in set(self._tokens(query)) if t in self.idf]
        out = []
        for c, tf, dl in zip(self.chunks, self.tf, self.len):
            if not keep(c, filters):
                continue
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avg))
                    for t in terms if t in tf)
            if s > 0:
                out.append({**c, 'score': s, 'strategy': 'bm25'})
        out.sort(key=lambda c: (-c['score'], c['id']))
        return [{**c, 'rank': i + 1} for i, c in enumerate(out[:top_k])]


class TfidfFiltered(TfidfIndex):
    def search(self, query, *, filters=None, top_k=3):
        res = [c for c in super().search(query, top_k=len(self.chunks)) if keep(c, filters)][:top_k]
        return [{**c, 'rank': i + 1} for i, c in enumerate(res)]


def rrf(ranked_lists, k=60, top_k=3):
    """Reciprocal Rank Fusion: rank-based, so dense cosine and BM25 scores need no normalization."""
    score, by_id = Counter(), {}
    for lst in ranked_lists:
        for rank, c in enumerate(lst, 1):
            score[c['id']] += 1 / (k + rank)
            by_id.setdefault(c['id'], c)
    ids = sorted(score, key=lambda i: (-score[i], i))[:top_k]
    return [{**by_id[i], 'score': score[i], 'rank': r, 'strategy': 'hybrid_rrf',
             'component_scores': {lst[0]['strategy']: next((c['score'] for c in lst if c['id'] == i), None)
                                  for lst in ranked_lists if lst}} for r, i in enumerate(ids, 1)]


class Hybrid:
    def __init__(self, dense, bm25, candidates=20, k=60):
        self.dense, self.bm25, self.candidates, self.k = dense, bm25, candidates, k

    def search(self, query, *, filters=None, top_k=3):
        return rrf([self.dense.search(query, filters=filters, top_k=self.candidates),
                    self.bm25.search(query, filters=filters, top_k=self.candidates)], k=self.k, top_k=top_k)


def evaluate(name, retriever, questions, *, use_filter, top_k, budget):
    traces = []
    for q in questions:
        filters = q.get('filter') if use_filter else None
        t0 = time.perf_counter()
        results = retriever.search(q['question'], filters=filters, top_k=top_k)
        ms = (time.perf_counter() - t0) * 1000
        context, ids = pack_context(results[:3], budget)
        marked = [{'id': c['id'], 'page': c['page'], 'rank': c['rank'], 'score': round(c['score'], 4),
                   'chapter': c['metadata'].get('chapter'), 'hit': hit(c, q['gold']),
                   **({'component_scores': c['component_scores']} if 'component_scores' in c else {})} for c in results]
        traces.append({'config': name, 'query_id': q['id'], 'type': q.get('type'), 'question': q['question'],
                       'filter_applied': filters, 'top_k': top_k, 'results': marked, 'context_ids': ids,
                       'context_chars': len(context), 'elapsed_ms': round(ms, 1),
                       'hit1': bool(marked and marked[0]['hit']), 'hit3': any(c['hit'] for c in marked[:3]),
                       'context_hit': any(c['hit'] and c['id'] in ids for c in marked)})
    return traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks', type=Path, default=Path('starter/data/chunks.json'))
    ap.add_argument('--questions', type=Path, default=Path('starter/data/questions.json'))
    ap.add_argument('--split', required=True, choices=('dev', 'test'))
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--budget', type=int, default=1500, help='context budget in characters (top-3 chunks ≤500 each)')
    args = ap.parse_args()
    from day6_store import DenseRetriever  # imported here so lexical-only tests need no DB

    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = [q for q in json.loads(args.questions.read_text(encoding='utf-8')) if q['split'] == args.split]
    validate(chunks, questions)
    dense, bm25 = DenseRetriever(), BM25(chunks)
    configs = {  # name: (retriever, use_filter, uses_embedding)
        'tfidf': (TfidfFiltered(chunks), False, False),
        'bm25': (bm25, False, False),
        'bm25+filter': (bm25, True, False),
        'dense': (dense, False, True),
        'dense+filter': (dense, True, True),
        'hybrid_rrf': (Hybrid(dense, bm25), False, True),
        'hybrid_rrf+filter': (Hybrid(dense, bm25), True, True),
    }
    all_traces, summary = [], {'split': args.split, 'n': len(questions), 'budget_chars': args.budget,
                               'embedding_model': dense.model, 'dimensions': dense.dimensions,
                               'day6_strategy': dense.strategy, 'configs': {}}
    for name, (retriever, use_filter, emb) in configs.items():
        tokens0 = dense.embed_tokens
        tr = evaluate(name, retriever, questions, use_filter=use_filter, top_k=3, budget=args.budget)
        tokens = dense.embed_tokens - tokens0 if emb else 0
        n = len(tr)
        summary['configs'][name] = {
            'hit1': f"{sum(t['hit1'] for t in tr)}/{n}", 'evidence_hit3': f"{sum(t['hit3'] for t in tr)}/{n}",
            'context_evidence': f"{sum(t['context_hit'] for t in tr)}/{n}",
            'hit1_rate': round(sum(t['hit1'] for t in tr) / n, 3), 'hit3_rate': round(sum(t['hit3'] for t in tr) / n, 3),
            'mrr@3': round(sum(next((1 / c['rank'] for c in t['results'][:3] if c['hit']), 0) for t in tr) / n, 3),
            'avg_context_chars': round(statistics.mean(t['context_chars'] for t in tr)),
            'median_ms': round(statistics.median(t['elapsed_ms'] for t in tr), 1),
            'embedding_tokens': tokens, 'cost_usd': round(tokens / 1e6 * PRICE_PER_M_TOKENS, 6)}
        all_traces += tr
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'traces.json').write_text(json.dumps(all_traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"{'config':<18} {'Hit@1':>6} {'Hit@3':>6} {'Ctx':>6} {'MRR':>5} {'ms':>7} {'$':>9}")
    for name, m in summary['configs'].items():
        print(f"{name:<18} {m['hit1']:>6} {m['evidence_hit3']:>6} {m['context_evidence']:>6} {m['mrr@3']:>5} "
              f"{m['median_ms']:>7} {m['cost_usd']:>9}")
    ids = [q['id'] for q in questions]
    print('\nper-question Hit@3 (1=hit):\n' + ' ' * 18 + ' '.join(i[-2:] for i in ids))
    for name in configs:
        row = {t['query_id']: t['hit3'] for t in all_traces if t['config'] == name}
        print(f"{name:<18} " + '  '.join('1' if row[i] else '.' for i in ids))


if __name__ == '__main__':
    main()
