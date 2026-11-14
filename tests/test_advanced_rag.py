"""Offline checks for advanced/rag.py retrieval helpers (no API or database call)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'advanced'))
import rag  # noqa: E402

D = rag.DOC


class AdvancedRagTests(unittest.TestCase):
    def test_label_lookup_returns_every_page_of_a_split_table(self):
        labels, pages = rag.label_chunks('In the stem-and-leaf plot of Table 2.85, which leaves are on stem 2?')
        self.assertEqual(labels, {'Table 2.85'})
        self.assertEqual(pages, [167, 168])

    def test_question_without_label_has_no_label_pages(self):
        self.assertEqual(rag.label_chunks('What is a chance experiment?'), (set(), []))

    def test_siblings_add_text_and_visual_of_same_page_once(self):
        out = rag.with_siblings([rag.BY_ID[f'{D}:p118:visual'], rag.BY_ID[f'{D}:p118:text']])
        self.assertEqual([c['id'] for c in out], [f'{D}:p118:visual', f'{D}:p118:text'])
        self.assertEqual([c['id'] for c in rag.with_siblings([rag.BY_ID[f'{D}:p22:text']])], [f'{D}:p22:text'])

    def test_pack_respects_budget(self):
        chunks = [rag.BY_ID[f'{D}:p22:text'], rag.BY_ID[f'{D}:p118:text']]
        context, ids = rag.pack(chunks, len(chunks[0]['text']) + 60)
        self.assertEqual(ids, [f'{D}:p22:text'])
        self.assertLessEqual(len(context), len(chunks[0]['text']) + 60)


if __name__ == '__main__':
    unittest.main()
