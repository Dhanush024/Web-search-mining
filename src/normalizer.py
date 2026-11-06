"""
Module 3: Text Normalization & Case Folding
==========================================
Performs linguistic normalization on tokens, including:
- Case folding (lowercasing)
- Diacritic / accent stripping
- Punctuation sanitization
- Optional stopword removal
- Built-in Porter Stemmer (suffix stripping without external dependencies)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Collection, Iterable, List, Optional, Set


# Standard English stop words commonly used in Information Retrieval
DEFAULT_STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class PorterStemmer:
    """
    Pure-Python implementation of the classic Martin Porter Stemming Algorithm (1980).
    Requires zero external dependencies.
    """

    def __init__(self) -> None:
        self.b = ""
        self.k = 0
        self.k0 = 0
        self.j = 0

    def _is_consonant(self, i: int) -> bool:
        char = self.b[i]
        if char in "aeiou":
            return False
        if char == "y":
            if i == self.k0:
                return True
            return not self._is_consonant(i - 1)
        return True

    def _measure(self) -> int:
        """Measures the number of consonant sequences between k0 and j."""
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self._is_consonant(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._is_consonant(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._is_consonant(i):
                    break
                i += 1
            i += 1

    def _vowel_in_stem(self) -> bool:
        for i in range(self.k0, self.j + 1):
            if not self._is_consonant(i):
                return True
        return False

    def _double_consonant(self, i: int) -> bool:
        if i < self.k0 + 1:
            return False
        if self.b[i] != self.b[i - 1]:
            return False
        return self._is_consonant(i)

    def _cvc(self, i: int) -> bool:
        if i < self.k0 + 2 or not self._is_consonant(i) or self._is_consonant(i - 1) or not self._is_consonant(i - 2):
            return False
        ch = self.b[i]
        if ch in "wxy":
            return False
        return True

    def _ends(self, s: str) -> bool:
        length = len(s)
        o = self.k - length + 1
        if o < self.k0:
            return False
        for i in range(length):
            if self.b[o + i] != s[i]:
                return False
        self.j = self.k - length
        return True

    def _set_to(self, s: str) -> None:
        length = len(s)
        o = self.j + 1
        self.b = self.b[:o] + s + self.b[o + length:]
        self.k = self.j + length

    def _replace(self, s: str) -> None:
        if self._measure() > 0:
            self._set_to(s)

    def _step1ab(self) -> None:
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._set_to("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1

        if self._ends("eed"):
            if self._measure() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowel_in_stem():
            self.k = self.j
            if self._ends("at"):
                self._set_to("ate")
            elif self._ends("bl"):
                self._set_to("ble")
            elif self._ends("iz"):
                self._set_to("ize")
            elif self._double_consonant(self.k):
                self.k -= 1
                ch = self.b[self.k]
                if ch in "lsz":
                    self.k += 1
            elif self._measure() == 1 and self._cvc(self.k):
                self._set_to("e")

    def _step1c(self) -> None:
        if self._ends("y") and self._vowel_in_stem():
            self.b = self.b[:self.k] + "i" + self.b[self.k + 1:]

    def _step2(self) -> None:
        pairs = {
            "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
            "izer": "ize", "bli": "ble", "alli": "al", "entli": "ent",
            "eli": "e", "ousli": "ous", "ization": "ize", "ation": "ate",
            "ator": "ate", "alism": "al", "iveness": "ive", "fulness": "ful",
            "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble"
        }
        for suffix, replacement in pairs.items():
            if self._ends(suffix):
                if self._measure() > 0:
                    self._set_to(replacement)
                break

    def _step3(self) -> None:
        pairs = {
            "icate": "ic", "ative": "", "alize": "al",
            "iciti": "ic", "ical": "ic", "ful": "", "ness": ""
        }
        for suffix, replacement in pairs.items():
            if self._ends(suffix):
                if self._measure() > 0:
                    self._set_to(replacement)
                break

    def _step4(self) -> None:
        suffixes = (
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
            "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize"
        )
        for suffix in suffixes:
            if self._ends(suffix):
                if self._measure() > 1:
                    self.k = self.j
                return
        if self._ends("ion") and self.j >= self.k0 and self.b[self.j] in "st":
            if self._measure() > 1:
                self.k = self.j

    def _step5(self) -> None:
        self.j = self.k
        if self.b[self.k] == "e":
            a = self._measure()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self._double_consonant(self.k) and self._measure() > 1:
            self.k -= 1

    def stem(self, word: str) -> str:
        """Stems a word according to the Porter stemmer rules."""
        word = word.lower()
        if len(word) <= 2:
            return word

        self.b = word
        self.k = len(word) - 1
        self.k0 = 0

        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[:self.k + 1]


class TextNormalizer:
    """
    Coordinates text normalization steps:
    - Case folding (lower)
    - Diacritic / accent stripping
    - Punctuation removal
    - Stop word filtering
    - Suffix stemming
    """

    def __init__(
        self,
        case_fold: bool = True,
        strip_accents: bool = True,
        remove_stopwords: bool = False,
        enable_stemming: bool = False,
        custom_stopwords: Optional[Collection[str]] = None,
    ):
        self.do_case_fold = case_fold
        self.do_strip_accents = strip_accents
        self.do_remove_stopwords = remove_stopwords
        self.do_stemming = enable_stemming

        self.stopwords = set(custom_stopwords) if custom_stopwords is not None else DEFAULT_STOP_WORDS
        self.stemmer = PorterStemmer()

    def normalize(self, token: str) -> Optional[str]:
        """
        Normalizes a single token. Returns None if the token is discarded (e.g. stopword or empty).

        Args:
            token: Input word/token string.

        Returns:
            Normalized token string or None if filtered out.
        """
        if not token:
            return None

        # 1. Unicode accent stripping (e.g. café -> cafe)
        if self.do_strip_accents:
            token = unicodedata.normalize("NFKD", token)
            token = "".join(ch for ch in token if not unicodedata.combining(ch))

        # 2. Case folding (lowercasing)
        if self.do_case_fold:
            token = token.lower()

        # 3. Strip any surrounding punctuation/symbols that clung to token
        token = token.strip(" \t\r\n'\"`.,;:!?()[]{}-_")

        if not token:
            return None

        # 4. Stop word filtering
        if self.do_remove_stopwords and token in self.stopwords:
            return None

        # 5. Suffix stemming
        if self.do_stemming:
            token = self.stemmer.stem(token)

        return token if token else None

    def normalize_tokens(self, tokens: Iterable[str]) -> List[str]:
        """
        Normalizes a sequence of tokens, filtering out empty and discarded tokens.

        Args:
            tokens: Iterable of token strings.

        Returns:
            List of normalized token strings.
        """
        normalized: List[str] = []
        for tok in tokens:
            norm = self.normalize(tok)
            if norm is not None:
                normalized.append(norm)
        return normalized

