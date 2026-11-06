"""
Unit tests for Module 1: Document Loading
"""

import tempfile
import unittest
from pathlib import Path

from src.document_loader import Document, DocumentLoader


class TestDocumentLoader(unittest.TestCase):

    def setUp(self):
        self.loader = DocumentLoader()

    def test_load_from_dict(self):
        docs_dict = {
            1: "Information retrieval and search engines.",
            2: "Web mining and data extraction.",
        }
        docs = self.loader.load_from_dict(docs_dict)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].doc_id, 1)
        self.assertIn("Information retrieval", docs[0].content)
        self.assertEqual(docs[1].doc_id, 2)

    def test_load_file_and_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            file1 = dir_path / "sample1.txt"
            file2 = dir_path / "sample2.txt"

            file1.write_text("Title One\nFirst document text content here.", encoding="utf-8")
            file2.write_text("Title Two\nSecond document content with text.", encoding="utf-8")

            docs = self.loader.load_directory(dir_path)
            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0].doc_id, 1)
            self.assertEqual(docs[1].doc_id, 2)
            self.assertEqual(docs[0].title, "Title One")
            self.assertEqual(docs[1].title, "Title Two")

    def test_snippet_generation(self):
        doc = Document(
            doc_id=1,
            title="Test Doc",
            content="This is a sentence. " * 20,
        )
        snip = doc.snippet(max_chars=50)
        self.assertTrue(len(snip) <= 53)
        self.assertTrue(snip.endswith("..."))


if __name__ == "__main__":
    unittest.main()

