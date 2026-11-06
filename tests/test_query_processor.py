"""
Unit tests for Module 5: Query Processing
"""

import unittest
from src.document_loader import Document
from src.inverted_index import InvertedIndex
from src.query_processor import QueryProcessor
from src.wildcard_processor import WildcardProcessor


class TestQueryProcessor(unittest.TestCase):

    def setUp(self):
        self.index = InvertedIndex()
        self.docs = [
            Document(doc_id=1, title="Doc 1", content="information retrieval in search engines"),
            Document(doc_id=2, title="Doc 2", content="web crawlers and web search"),
            Document(doc_id=3, title="Doc 3", content="data mining and text mining techniques"),
            Document(doc_id=4, title="Doc 4", content="inverted index for search retrieval"),
        ]
        self.index.build(self.docs)
        self.wp = WildcardProcessor()
        self.qp = QueryProcessor(index=self.index, wildcard_processor=self.wp)

    def test_boolean_and(self):
        res = self.qp.search_boolean("search AND retrieval")
        doc_ids = [r.doc_id for r in res]
        self.assertEqual(sorted(doc_ids), [1, 4])

    def test_boolean_or(self):
        res = self.qp.search_boolean("crawlers OR mining")
        doc_ids = [r.doc_id for r in res]
        self.assertEqual(sorted(doc_ids), [2, 3])

    def test_boolean_not(self):
        res = self.qp.search_boolean("search NOT retrieval")
        doc_ids = [r.doc_id for r in res]
        self.assertEqual(doc_ids, [2])

    def test_exact_phrase_query(self):
        res = self.qp.search_phrase("search engines")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].doc_id, 1)

    def test_phrase_query_not_consecutive(self):
        # 'information' and 'engines' are both in doc 1, but not adjacent
        res = self.qp.search_phrase("information engines")
        self.assertEqual(len(res), 0)

    def test_wildcard_in_boolean(self):
        res = self.qp.search_boolean("retriev* AND search")
        doc_ids = [r.doc_id for r in res]
        self.assertEqual(sorted(doc_ids), [1, 4])

    def test_ranked_search(self):
        res = self.qp.search_ranked("data mining")
        self.assertTrue(len(res) > 0)
        self.assertEqual(res[0].doc_id, 3)


if __name__ == "__main__":
    unittest.main()

