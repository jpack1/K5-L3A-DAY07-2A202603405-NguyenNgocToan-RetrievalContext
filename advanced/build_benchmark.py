"""Check advanced/benchmark/questions.json and write the agent inputs (data/cases.json, data/questions.json).

Run from repo root: python advanced/build_benchmark.py
Checks: ids unique, split/group minimums from ADVANCED.md, every text quote appears verbatim (whitespace-normalized)
on its PDF page, every required page is indexed, and every 'absent_terms' entry of an unanswerable question has
zero hits in the indexed text (or on 'absent_on_pages' only). Gold answers stay out of cases.json (the only file the agent reads before answering).
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN = {'dev': {'table': 3, 'text': 3, 'chart': 3, 'visual': 3, 'multihop': 3, 'unanswerable': 3},
       'test': {'table': 6, 'text': 4, 'chart': 5, 'visual': 5, 'multihop': 4, 'unanswerable': 6}}
norm = lambda s: ' '.join(s.split())


def main():
    qs = json.loads((ROOT / 'benchmark' / 'questions.json').read_text(encoding='utf-8'))
    pages = {p['page']: norm(p['text']) for p in map(json.loads, open(ROOT / 'data' / 'pages.jsonl', encoding='utf-8'))}
    corpus = ' '.join(pages.values()).lower()
    assert len({q['id'] for q in qs}) == len(qs), 'duplicate ids'
    for q in qs:
        assert q['split'] in ('dev', 'dev2', 'test') and q['group'] in MIN['test'], q['id']
        if q['group'] == 'unanswerable':
            assert q['absent_terms'] and q['check'], q['id']
            # 'absent_on_pages': a value missing from one table, while the term may occur elsewhere in the book.
            scope = ' '.join(pages[n] for n in q['absent_on_pages']).lower() if 'absent_on_pages' in q else corpus
            for term in q['absent_terms']:
                assert scope.count(term.lower()) == 0, f"{q['id']}: '{term}' occurs in the checked text"
            continue
        assert q['expected_answer'] and q['required_pages'] and q['evidence'], q['id']
        assert set(q['required_pages']) <= set(pages), f"{q['id']}: page not indexed"
        assert len(q['required_pages']) >= 2 or q['group'] != 'multihop', f"{q['id']}: multi-hop needs two pages"
        for e in q['evidence']:
            assert e['page'] in q['required_pages'], f"{q['id']}: evidence page {e['page']} not required"
            if 'quote' in e:
                assert norm(e['quote']) in pages[e['page']], f"{q['id']}: quote not on p{e['page']}: {e['quote']!r}"
            else:
                assert e['kind'] in ('chart', 'diagram') and e['locator'] and e['value'], q['id']
    for split, need in MIN.items():
        have = Counter(q['group'] for q in qs if q['split'] == split)
        for g, n in need.items():
            assert have[g] >= n, f'{split} {g}: {have[g]} < {n}'
        print(split, dict(have), 'answerable', sum(v for g, v in have.items() if g != 'unanswerable'))
    cases = [{'id': q['id'], 'split': q['split'], 'question': q['question'], 'filter': None} for q in qs]
    (ROOT / 'data' / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding='utf-8')
    # The agent script only reads this after all answers are produced, for context-evidence metrics.
    gold = [{'id': q['id'], 'required_pages': q['required_pages']} for q in qs if q['group'] != 'unanswerable']
    (ROOT / 'data' / 'questions.json').write_text(json.dumps(gold, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'benchmark OK: {len(qs)} questions')


if __name__ == '__main__':
    main()
