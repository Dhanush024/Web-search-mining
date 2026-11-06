"""
Module 6: Wildcard Processing
=============================
Implements classic Information Retrieval wildcard query algorithms:
1. Permuterm Index: Rotates vocabulary terms around a sentinel '$' to
   convert leading, trailing, and middle wildcards into prefix lookups.
2. K-Gram (Character N-Gram) Index: Decomposes vocabulary into k-character
   slices for fast substring and complex wildcard matching.
3. Post-filtering / Regex matching for arbitrary wildcards (e.g. '?', multiple '*').
"""

from __future__ import annotations

import fnmatch
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple


class PermutermIndex:
    """
    Permuterm Index structure for resolving wildcard queries.

    For a term 'word', appends sentinel '$' ('word$') and stores all cyclic rotations:
    'word$', 'ord$w', 'rd$wo', 'd$wor', '$word' -> 'word'
    """

    SENTINEL = "$"

    def __init__(self) -> None:
        # Maps rotation string -> set of original terms
        self.rotations: Dict[str, Set[str]] = defaultdict(set)
        # Sorted list of rotations for fast binary search prefix lookups
        self._sorted_rotations: List[Tuple[str, str]] = []
        self._is_dirty = True

    def add_term(self, term: str) -> None:
        """
        Generates and indexes all cyclic permutations of term + '$'.
        """
        if not term:
            return
        augmented = term + self.SENTINEL
        length = len(augmented)
        for i in range(length):
            rotation = augmented[i:] + augmented[:i]
            self.rotations[rotation].add(term)
        self._is_dirty = True

    def build(self, vocabulary: Iterable[str]) -> None:
        """Builds permuterm index from an iterable of terms."""
        self.rotations.clear()
        for term in vocabulary:
            self.add_term(term)
        self._finalize()

    def _finalize(self) -> None:
        """Flattens and sorts rotations for fast prefix scanning."""
        items: List[Tuple[str, str]] = []
        for rotation, terms in self.rotations.items():
            for t in terms:
                items.append((rotation, t))
        items.sort(key=lambda x: x[0])
        self._sorted_rotations = items
        self._is_dirty = False

    def query_prefix(self, prefix: str) -> Set[str]:
        """
        Finds all original terms whose rotation begins with the given prefix.
        """
        if self._is_dirty:
            self._finalize()

        matched: Set[str] = set()
        # Binary search or fast prefix scan across sorted rotations
        for rotation, term in self._sorted_rotations:
            if rotation.startswith(prefix):
                matched.add(term)
            elif rotation > prefix and not prefix.startswith(rotation[:len(prefix)]):
                # Optimization: Since items are sorted, we can break once past prefix range
                if rotation[:len(prefix)] > prefix:
                    break
        return matched

    def rotate_query(self, pattern: str) -> Tuple[str, bool]:
        """
        Rotates a wildcard pattern so that the '*' appears at the end.
        Returns (prefix_to_lookup, is_exact_or_prefix).

        Rules:
        - X*   -> lookup $X
        - *X   -> lookup X$
        - X*Y  -> lookup Y$X
        - *X*  -> lookup X (then post-filter)
        """
        augmented = pattern + self.SENTINEL
        parts = augmented.split("*")

        if len(parts) == 1:
            # No wildcard, exact match: word$
            return augmented, True
        elif len(parts) == 2:
            # Single wildcard: parts[0] * parts[1]
            # e.g., 'X*Y$' -> rotate so '*' is at the end -> 'Y$X*'
            prefix = parts[1] + parts[0]
            return prefix, False
        else:
            # Multiple wildcards: take the segment with '$' as anchor
            # e.g., 'X*Y*Z$' -> parts are [X, Y, Z$]
            # anchor is Z$X
            anchor = parts[-1] + parts[0]
            return anchor, False


class KGramIndex:
    """
    K-Gram (Character N-gram) index for substring and multi-wildcard matching.
    Default k=2 (bi-gram) and k=3 (tri-gram).
    """

    SENTINEL = "$"

    def __init__(self, k: int = 2):
        self.k = k
        self.index: Dict[str, Set[str]] = defaultdict(set)

    def add_term(self, term: str) -> None:
        augmented = f"{self.SENTINEL}{term}{self.SENTINEL}"
        for i in range(len(augmented) - self.k + 1):
            gram = augmented[i:i + self.k]
            self.index[gram].add(term)

    def build(self, vocabulary: Iterable[str]) -> None:
        self.index.clear()
        for term in vocabulary:
            self.add_term(term)

    def get_terms_for_gram(self, gram: str) -> Set[str]:
        return self.index.get(gram, set())


class WildcardProcessor:
    """
    Unified Wildcard Query Processor combining Permuterm Index,
    K-gram filtering, and regex post-validation.
    """

    def __init__(self, vocabulary: Optional[Iterable[str]] = None):
        self.vocabulary: Set[str] = set()
        self.permuterm = PermutermIndex()
        self.bigram = KGramIndex(k=2)

        if vocabulary:
            self.build(vocabulary)

    def build(self, vocabulary: Iterable[str]) -> None:
        """Indexes vocabulary terms for wildcard processing."""
        self.vocabulary = set(vocabulary)
        self.permuterm.build(self.vocabulary)
        self.bigram.build(self.vocabulary)

    def has_wildcard(self, term: str) -> bool:
        """Returns True if the term contains wildcard characters (* or ?)."""
        return "*" in term or "?" in term

    def _wildcard_to_regex(self, pattern: str) -> re.Pattern[str]:
        """Converts standard glob wildcard pattern (*, ?) to compiled regex."""
        # fnmatch.translate produces e.g. '(?s:pattern)\\Z'
        regex_str = fnmatch.translate(pattern)
        return re.compile(regex_str, re.IGNORECASE)

    def resolve(self, wildcard_pattern: str) -> List[str]:
        """
        Resolves a wildcard query pattern against the indexed vocabulary.

        Examples:
            'retriev*'   -> ['retrieval', 'retrieved', 'retrieving']
            '*mining'    -> ['mining', 'datamining']
            'cr*wler'    -> ['crawler']
            'inf?rmation'-> ['information']

        Args:
            wildcard_pattern: Search token containing '*' or '?'.

        Returns:
            Sorted list of matching vocabulary terms.
        """
        clean_pattern = wildcard_pattern.strip().lower()
        if not clean_pattern:
            return []

        if not self.has_wildcard(clean_pattern):
            # Plain word lookup
            return [clean_pattern] if clean_pattern in self.vocabulary else []

        regex = self._wildcard_to_regex(clean_pattern)

        # 1. Try Permuterm Index if single asterisk and no '?'
        if clean_pattern.count("*") == 1 and "?" not in clean_pattern:
            prefix, is_exact = self.permuterm.rotate_query(clean_pattern)
            candidates = self.permuterm.query_prefix(prefix)
            # Post-filter to ensure strict fidelity
            matched = [t for t in candidates if regex.match(t)]
            return sorted(matched)

        # 2. General case (multiple '*' or '?'):
        # Extract longest non-wildcard fragments for k-gram candidate pruning
        fragments = [frag for frag in re.split(r"[*?]+", clean_pattern) if len(frag) >= 2]

        if fragments:
            candidate_sets: List[Set[str]] = []
            for frag in fragments:
                frag_candidates: Optional[Set[str]] = None
                for i in range(len(frag) - 1):
                    bi = frag[i:i + 2]
                    terms_with_bi = self.bigram.get_terms_for_gram(bi)
                    if frag_candidates is None:
                        frag_candidates = set(terms_with_bi)
                    else:
                        frag_candidates &= terms_with_bi
                if frag_candidates is not None:
                    candidate_sets.append(frag_candidates)

            if candidate_sets:
                # Intersect candidate sets from fragments
                candidates = set.intersection(*candidate_sets)
            else:
                candidates = self.vocabulary
        else:
            candidates = self.vocabulary

        # Validate candidates against full regex
        matched = [t for t in candidates if regex.match(t)]
        return sorted(matched)

