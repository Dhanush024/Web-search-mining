"""
Unit tests for Module 2: Tokenization
"""

import unittest
from src.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_basic_tokenization(self):
        text = "Information Retrieval (IR) is very important!"
        tokens = self.tokenizer.tokenize(text)
        expected = ["Information", "Retrieval", "IR", "is", "very", "important"]
        self.assertEqual(tokens, expected)

    def test_token_with_positions(self):
        text = "Web search and Web mining"
        pos_tokens = self.tokenizer.tokenize_with_positions(text)
        self.assertEqual(len(pos_tokens), 5)
        self.assertEqual(pos_tokens[0].text, "Web")
        self.assertEqual(pos_tokens[0].position, 0)
        self.assertEqual(pos_tokens[1].text, "search")
        self.assertEqual(pos_tokens[1].position, 1)
        self.assertEqual(pos_tokens[3].text, "Web")
        self.assertEqual(pos_tokens[3].position, 3)

    def test_empty_string(self):
        self.assertEqual(self.tokenizer.tokenize(""), [])
        self.assertEqual(self.tokenizer.tokenize_with_positions(""), [])


if __name__ == "__main__":
    unittest.main()

