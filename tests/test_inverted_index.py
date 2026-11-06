"""
Unit tests for Module 4: Inverted Index Construction
"""

import tempfile
import unittest
from pathlib import Path

from src.document_loader import Document
from src.inverted_index import InvertedIndex


class TestInvertedIndex(unittest.TestCase):

    def setUp(self):
        self.index = InvertedIndex()
        self.docs = [
            Document(doc_id=1, title="Doc 1", content="information retrieval system"),
            Document(doc_id=2, title="Doc 2", content="web search and retrieval engine"),
            Document(doc_id=3, title="Doc 3", content="search engine inverted index"),
        ]
        self.index.build(self.docs)

    def test_postings_retrieval(self):
        postings = self.index.get_postings("retrieval")
        doc_ids = [p.doc_id for p in postings]
        self.assertEqual(doc_ids, [1, 2])
        self.assertEqual(self.index.get_doc_frequency("retrieval"), 2)

    def test_term_positions(self):
        postings = self.index.get_postings("search")
        doc3_posting = next(p for p in postings if p.doc_id == 3)
        self.assertIn(0, doc3_posting.positions)

    def test_vocabulary(self):
        vocab = self.index.get_vocabulary()
        self.assertIn("information", vocab)
        self.assertIn("inverted", vocab)
        self.assertIn("engine", vocab)

    def test_json_serialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_index.json"
            self.index.save_to_json(file_path)

            loaded_index = InvertedIndex()
            loaded_index.load_from_json(file_path)

            self.assertEqual(loaded_index.total_documents, 3)
            self.assertEqual(loaded_index.get_doc_frequency("retrieval"), 2)
            self.assertEqual(loaded_index.get_postings("index")[0].doc_id, 3)


if __name__ == "__main__":
    unittest.main()

