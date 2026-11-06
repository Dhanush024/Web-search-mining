"""
Unit tests for Module 6: Wildcard Processing (Permuterm and K-Gram Indexing)
"""

import unittest
from src.wildcard_processor import KGramIndex, PermutermIndex, WildcardProcessor


class TestWildcardProcessor(unittest.TestCase):

    def setUp(self):
        self.vocab = [
            "search", "searching", "searched", "crawler", "crawling",
            "mining", "datamining", "retrieval", "retriever", "index",
            "indexes", "indexing", "vector", "processor", "processing"
        ]
        self.wp = WildcardProcessor(self.vocab)

    def test_permuterm_rotations(self):
        permuterm = PermutermIndex()
        permuterm.add_term("hello")
        rotations = list(permuterm.rotations.keys())
        self.assertIn("hello$", rotations)
        self.assertIn("ello$h", rotations)
        self.assertIn("$hello", rotations)

    def test_trailing_wildcard(self):
        matched = self.wp.resolve("search*")
        self.assertIn("search", matched)
        self.assertIn("searched", matched)
        self.assertIn("searching", matched)
        self.assertNotIn("crawler", matched)

    def test_leading_wildcard(self):
        matched = self.wp.resolve("*mining")
        self.assertIn("mining", matched)
        self.assertIn("datamining", matched)
        self.assertNotIn("crawler", matched)

    def test_middle_wildcard(self):
        matched = self.wp.resolve("cr*er")
        self.assertIn("crawler", matched)
        self.assertNotIn("mining", matched)

    def test_question_mark_wildcard(self):
        matched = self.wp.resolve("inde?")
        self.assertIn("index", matched)
        self.assertNotIn("indexes", matched)

    def test_multiple_wildcards(self):
        matched = self.wp.resolve("*ind*")
        self.assertIn("index", matched)
        self.assertIn("indexes", matched)
        self.assertIn("indexing", matched)


if __name__ == "__main__":
    unittest.main()

