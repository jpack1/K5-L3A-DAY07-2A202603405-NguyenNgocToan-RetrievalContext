"""Compare agent runs and export manual-review sheets.

From starter/:
    PYTHONUTF8=1 python compare.py results/exp-dev/R*/            # print comparison table
    PYTHONUTF8=1 python compare.py results/exp-dev/R0*/ --review   # also write <run>/review.csv to fill by hand
    PYTHONUTF8=1 python compare.py results/exp-dev/R0*/ --to-json  # review.csv -> reviews.json for agent_metrics.py
Review columns: answer_grade 0/1/2 (answerable), citation_support one 1/0 per citation separated by ';',
abstention_correct 1/0 (unanswerable). Draft values must be checked by the student.
"""
import argparse
import csv
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+', type=Path)
ap.add_argument('--review', action='store_true')
ap.add_argument('--to-json', action='store_true')
a = ap.parse_args()

questions = {q['id']: q for q in json.loads(Path('data/questions.json').read_text(encoding='utf-8'))}
chunks = {c['id']: c for c in json.loads(Path('data/chunks.json').read_text(encoding='utf-8'))}
cols = ['run', 'tool_use', 'searches_distribution', 'context_evidence_rate', 'false_abstention',
        'abstention_unanswerable', 'answers_with_citation', 'median_s', 'total_cost_usd', 'avg_context_chars']
print(' | '.join(cols))
for run in a.runs:
    s = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    traces = json.loads((run / 'traces.json').read_text(encoding='utf-8'))
    print(' | '.join([run.name] + [str(s[c]) for c in cols[1:]]))
    if a.review:
        out = run / 'review.csv'
        if out.exists():
            raise SystemExit(f'{out} exists; not overwriting manual labels')
        with open(out, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['query_id', 'question', 'expected_answer', 'status', 'answer', 'citations', 'cited_text',
                        'context_has_gold', 'answer_grade', 'citation_support', 'abstention_correct', 'reason', 'grader'])
            for t in traces:
                q = questions.get(t['query_id'], {})
                cited = '\n---\n'.join(f"[{i}] " + ' '.join(chunks[i]['text'].split()) for i in t['citations'] if i in chunks)
                w.writerow([t['query_id'], t['question'], q.get('expected_answer', '(unanswerable)'), t['status'],
                            t.get('answer') or t.get('error'), ';'.join(t['citations']), cited,
                            t.get('context_evidence', ''), '', '', '', '', ''])
        print(f'  wrote {out}')
    if a.to_json:
        rows = list(csv.DictReader(open(run / 'review.csv', encoding='utf-8-sig')))
        reviews = []
        for r in rows:
            rv = {'query_id': r['query_id'],
                  'citation_support': [x.strip() == '1' for x in r['citation_support'].split(';') if x.strip()]}
            if r['query_id'] in questions:
                rv['answer_grade'] = int(r['answer_grade'])
            else:
                rv['abstention_correct'] = r['abstention_correct'].strip() == '1'
            reviews.append(rv)
        (run / 'reviews.json').write_text(json.dumps(reviews, indent=2) + '\n', encoding='utf-8')
        print(f'  wrote {run / "reviews.json"}')
