"""Ask the agent one question and print what it did.

From starter/:  PYTHONUTF8=1 python ask.py "How do I leave an infinite loop?" [--chapter 5] [--method dense]
"""
import argparse
import json
from pathlib import Path

from agent import OpenAIController, build_retriever
from agentic import run_agent

ap = argparse.ArgumentParser()
ap.add_argument('question')
ap.add_argument('--method', default='hybrid_rrf', choices=('dense', 'bm25', 'hybrid_rrf'))
ap.add_argument('--chapter', type=int, help='optional user-provided filter: 5, 9 or 10')
ap.add_argument('--budget', type=int, default=3000)
ap.add_argument('--max-searches', type=int, default=3)
a = ap.parse_args()

chunks = json.loads(Path('data/chunks.json').read_text(encoding='utf-8'))
_, retriever = build_retriever(a.method, chunks)
ctl = OpenAIController('gpt-4o-mini', a.max_searches)
r = run_agent(a.question, ctl, retriever, filters={'chapter': a.chapter} if a.chapter else None,
              max_searches=a.max_searches, top_k=3, budget=a.budget)
for i, e in enumerate((e for e in r['events'] if e['event'] == 'search'), 1):
    print(f"search {i}: {e['query']!r}  filter={e['filter']}")
    for c in e['results']:
        print(f"   {c['rank']}. {c['score']:.3f} p{c['page']} {c['id']} | {' '.join(c['text'].split())[:80]}")
print(f"\nstatus: {r['status']}\nanswer: {r['answer']}\ncitations: {r['citations']}")
