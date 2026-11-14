"""Offline checks for BM25, metadata filter and RRF fusion (no DB/API)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from retrieval import BM25, rrf


def chunk(id, text, chapter):
    return {'id': id, 'source_id': 'd', 'source': 'd.pdf', 'page': 1, 'text': text, 'metadata': {'chapter': chapter}}


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [chunk('a', 'pop removes the last list element', 9),
                       chunk('b', 'pop removes the dictionary item with the key', 10),
                       chunk('c', 'a while loop repeats', 5)]

    def test_bm25_ranks_and_filters(self):
        bm = BM25(self.chunks)
        self.assertEqual(bm.search('dictionary pop key')[0]['id'], 'b')
        res = bm.search('dictionary pop key', filters={'chapter': 9})
        self.assertEqual([c['id'] for c in res], ['a'])
        self.assertEqual(bm.search('unrelated zebra'), [])

    def test_bm25_bigrams_match_data_sequence(self):
        chunks = [chunk('x', 'data 4; 5; 6; 6; 7; 7 shown', 2), chunk('y', '7 6 5 4 numbers again 6 7', 2),
                  chunk('z', 'other text', 2)]
        self.assertEqual(BM25(chunks, bigrams=True).search('4; 5; 6; 6; 7')[0]['id'], 'x')

    def test_rrf_rewards_agreement(self):
        d = [{'id': 'x', 'score': .9, 'strategy': 'dense'}, {'id': 'y', 'score': .8, 'strategy': 'dense'}]
        l = [{'id': 'y', 'score': 5, 'strategy': 'bm25'}, {'id': 'z', 'score': 4, 'strategy': 'bm25'}]
        out = rrf([d, l], k=60, top_k=3)
        self.assertEqual([c['id'] for c in out], ['y', 'x', 'z'])
        self.assertEqual(out[0]['component_scores'], {'dense': .8, 'bm25': 5})


if __name__ == '__main__':
    unittest.main()
