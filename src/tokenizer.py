"""
Module 2: Tokenization
======================
Extracts lexical tokens from raw character streams, with optional tracking
of word positions and character offsets for positional inverted indexes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Pattern


@dataclass(frozen=True)
class Token:
    """
    Represents an extracted token with positional information.

    Attributes:
        text: The string literal of the token.
        position: 0-indexed word position within the document stream.
        start_char: Starting character index in the source text.
        end_char: Ending character index in the source text.
    """
    text: str
    position: int
    start_char: int
    end_char: int

    def __repr__(self) -> str:
        return f"Token('{self.text}', pos={self.position})"


class Tokenizer:
    """
    Tokenizes raw text into individual terms, tracking positional metadata.
    """

    # Default pattern: sequences of alphanumeric characters, supporting internal apostrophes/hyphens
    DEFAULT_PATTERN = r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?"

    def __init__(self, pattern: str = DEFAULT_PATTERN):
        self.pattern_str = pattern
        self.regex: Pattern[str] = re.compile(pattern)

    def tokenize_with_positions(self, text: str) -> List[Token]:
        """
        Tokenizes text into a list of Token objects, each recording
        its word position index and character offsets.

        Args:
            text: Raw input text.

        Returns:
            List of Token instances.
        """
        if not text:
            return []

        tokens: List[Token] = []
        for word_pos, match in enumerate(self.regex.finditer(text)):
            tok_text = match.group(0)
            tokens.append(
                Token(
                    text=tok_text,
                    position=word_pos,
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )
        return tokens

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text into a simple list of string tokens.

        Args:
            text: Raw input text.

        Returns:
            List of token strings.
        """
        if not text:
            return []
        return self.regex.findall(text)

