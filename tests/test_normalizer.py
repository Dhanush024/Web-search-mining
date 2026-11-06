"""
Unit tests for Module 3: Text Normalization and Case Folding
"""

import unittest
from src.normalizer import PorterStemmer, TextNormalizer


class TestNormalizer(unittest.TestCase):

    def test_case_folding(self):
        norm = TextNormalizer(case_fold=True, strip_accents=False, remove_stopwords=False)
        self.assertEqual(norm.normalize("Search"), "search")
        self.assertEqual(norm.normalize("ENGINE"), "engine")

    def test_accent_stripping(self):
        norm = TextNormalizer(case_fold=True, strip_accents=True)
        self.assertEqual(norm.normalize("résumé"), "resume")
        self.assertEqual(norm.normalize("CAFÉ"), "cafe")

    def test_stopword_removal(self):
        norm = TextNormalizer(case_fold=True, remove_stopwords=True)
        self.assertIsNone(norm.normalize("the"))
        self.assertIsNone(norm.normalize("is"))
        self.assertEqual(norm.normalize("retrieval"), "retrieval")

    def test_porter_stemmer(self):
        stemmer = PorterStemmer()
        self.assertEqual(stemmer.stem("crawlers"), "crawler")
        self.assertEqual(stemmer.stem("retrieving"), "retriev")
        self.assertEqual(stemmer.stem("mining"), "mine")

    def test_normalization_pipeline(self):
        norm = TextNormalizer(case_fold=True, strip_accents=True, remove_stopwords=True)
        tokens = ["The", "Quick", "BROWN", "Fox", "jumps", "over", "a", "lazy", "dog"]
        normalized = norm.normalize_tokens(tokens)
        self.assertNotIn("the", normalized)
        self.assertNotIn("a", normalized)
        self.assertIn("quick", normalized)
        self.assertIn("brown", normalized)
        self.assertIn("fox", normalized)


if __name__ == "__main__":
    unittest.main()

