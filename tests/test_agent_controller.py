"""Offline checks for OpenAIController normalization; fake client, no API call or quality claim."""
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY') or 'test-only'  # also when .env sets it empty
from agent import OpenAIController, PairPageText


class FakeClient:
    def __init__(self, name, args):
        self.reply, self.kwargs = (name, args), None
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.kwargs = kwargs
        call = NS(function=NS(name=self.reply[0], arguments=json.dumps(self.reply[1])))
        return NS(choices=[NS(message=NS(tool_calls=[call]))], usage=NS(prompt_tokens=1, completion_tokens=1))


def controller(name, args):
    c = OpenAIController('gpt-4o-mini', 3)
    c.client = FakeClient(name, args)
    return c


class ControllerTests(unittest.TestCase):
    def test_first_step_forces_search(self):
        c = controller('search', {'query': 'loop'})
        self.assertEqual(c.decide('q', []), {'action': 'search', 'query': 'loop'})
        self.assertEqual(c.client.kwargs['tool_choice']['function']['name'], 'search')

    def test_missing_inline_markers_are_appended_and_logged(self):
        c = controller('answer', {'answer': 'Yes.', 'citations': ['c1']})
        out = c.decide('q', [{'queries': ['x'], 'context': 'ctx'}])
        self.assertEqual(out['answer'], 'Yes. [c1]')
        self.assertEqual(c.calls[-1]['citation_markers_appended'], ['c1'])
        self.assertEqual(c.client.kwargs['tool_choice'], 'required')

    def test_repeat_query_stops_only_when_enabled(self):
        h = [{'queries': ['Loop  Else'], 'context': 'ctx'}]
        c = controller('search', {'query': 'loop else'})
        self.assertEqual(c.decide('q', h)['action'], 'search')
        c = controller('search', {'query': 'loop else'}); c.stop_repeat = True
        self.assertEqual(c.decide('q', h), {'action': 'answer', 'answer': '', 'citations': []})
        self.assertTrue(c.calls[-1]['repeat_query_stop'])

    def test_rank_context_order_keeps_better_later_chunk(self):
        from agentic import run_agent
        a = {'id': 'a', 'text': 'x' * 90, 'source_id': 'd', 'source': 's', 'page': 1, 'metadata': {}, 'rank': 2}
        b = {**a, 'id': 'b', 'rank': 1}
        acts = iter([{'action': 'search', 'query': '1'}, {'action': 'search', 'query': '2'},
                     {'action': 'answer', 'answer': 'ok [b]', 'citations': ['b']}])
        ctl = NS(decide=lambda q, h: next(acts))
        seq = iter([[a], [b]])
        ret = NS(search=lambda q, filters, top_k: next(seq))
        out = run_agent('q', ctl, ret, budget=130, context_order='rank')
        self.assertEqual(out['status'], 'answered')  # with 'arrival', b would not fit and the answer is rejected

    def test_first_search_can_use_original_question_without_model_call(self):
        c = controller('abstain', {'reason': 'x'}); c.first_original = True
        self.assertEqual(c.decide('raw question?', []), {'action': 'search', 'query': 'raw question?'})
        self.assertIsNone(c.client.kwargs)  # no API call made

    def test_abstain_before_min_searches_forces_another_search(self):
        replies = iter([('abstain', {'reason': 'x'}), ('search', {'query': 'other words'})])
        c = controller('abstain', {}); c.min_searches = 2
        def create(**kw):
            name, args = next(replies); c.client.kwargs = kw
            call = NS(function=NS(name=name, arguments=json.dumps(args)))
            return NS(choices=[NS(message=NS(tool_calls=[call]))], usage=NS(prompt_tokens=1, completion_tokens=1))
        c.client.chat.completions.create = create
        out = c.decide('q', [{'queries': ['first'], 'context': 'ctx'}])
        self.assertEqual(out, {'action': 'search', 'query': 'other words'})
        self.assertTrue(c.calls[0]['abstain_rejected'])
        self.assertEqual(c.client.kwargs['tool_choice']['function']['name'], 'search')

    def test_abstain_accepted_once_min_searches_reached(self):
        c = controller('abstain', {'reason': 'x'}); c.min_searches = 2
        self.assertEqual(c.decide('q', [{'queries': ['a', 'b'], 'context': ''}])['citations'], [])

    def test_fix_citations_adds_chunk_holding_numbers_only(self):
        from agent import fix_citations
        by_id = {'a': {'text': 'Installation F before 8 after 0, total 1,253'}, 'b': {'text': 'unrelated page'},
                 'c': {'text': 'definition text'}}
        ans, cites, log = fix_citations('F went from 1,253 to 0 [b] [c]', ['b', 'c'], ['b', 'a', 'c'], by_id)
        self.assertEqual(cites, ['b', 'c', 'a'])  # added, never removed
        self.assertTrue(ans.endswith('[a]'))
        self.assertEqual(log['unsupported_numbers'], [])
        self.assertEqual(fix_citations('no numbers here [c]', ['c'], ['a'], by_id)[2], None)

    def test_repeat_query_with_repeat_answer_decides_from_context(self):
        replies = iter([('search', {'query': 'same'}), ('answer', {'answer': 'X.', 'citations': ['c1']})])
        c = controller('search', {}); c.stop_repeat = True; c.repeat_answer = True
        def create(**kw):
            name, args = next(replies); c.client.kwargs = kw
            call = NS(function=NS(name=name, arguments=json.dumps(args)))
            return NS(choices=[NS(message=NS(tool_calls=[call]))], usage=NS(prompt_tokens=1, completion_tokens=1))
        c.client.chat.completions.create = create
        out = c.decide('q', [{'queries': ['same'], 'context': 'ctx'}])
        self.assertEqual(out['citations'], ['c1'])
        self.assertNotIn('search', [t['function']['name'] for t in c.client.kwargs['tools']])

    def test_drop_invalid_citations_keeps_valid_answer(self):
        from agentic import run_agent
        a = {'id': 'a', 'text': 'fact', 'source_id': 'd', 'source': 's', 'page': 1, 'metadata': {}, 'rank': 1}
        for drop, status in ((False, 'abstain'), (True, 'answered')):
            acts = iter([{'action': 'search', 'query': 'q'}, {'action': 'answer', 'answer': 'F [a] [zz]', 'citations': ['a', 'zz']}])
            out = run_agent('q', NS(decide=lambda q, h: next(acts)), NS(search=lambda q, filters, top_k: [a]),
                            budget=500, drop_invalid_citations=drop)
            self.assertEqual(out['status'], status)
        self.assertEqual(out['citations'], ['a'])

    def test_abstain_maps_to_empty_citations(self):
        c = controller('abstain', {'reason': 'none'})
        self.assertEqual(c.decide('q', [{'queries': ['x'], 'context': ''}])['citations'], [])


class PairPageTextTests(unittest.TestCase):
    def test_visual_hit_brings_page_text_and_keeps_top_k(self):
        by_id = {i: {'id': i, 'text': i} for i in ('d:p1:visual', 'd:p1:text', 'd:p2:text', 'd:p3:text')}
        inner = NS(search=lambda q, filters=None, top_k=3: [
            {**by_id['d:p1:visual'], 'rank': 1}, {**by_id['d:p2:text'], 'rank': 2}, {**by_id['d:p1:text'], 'rank': 3}])
        out = PairPageText(inner, by_id).search('q', top_k=3)
        self.assertEqual([(c['id'], c['rank']) for c in out], [('d:p1:visual', 1), ('d:p1:text', 1), ('d:p2:text', 2)])
        self.assertEqual(len(PairPageText(inner, by_id).search('q', top_k=2)), 2)


if __name__ == '__main__':
    unittest.main()
