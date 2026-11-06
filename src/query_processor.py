"""
Module 5: Query Processing
==========================
Evaluates search queries against the Inverted Index, supporting:
1. Boolean queries with AND, OR, NOT operations (postings list intersection, union, difference)
2. Phrase queries with positional postings verification (pos_{i+1} == pos_i + 1)
3. Wildcard query expansion via WildcardProcessor (Permuterm / K-gram)
4. Ranked free-text retrieval using TF-IDF and Cosine Similarity
5. Rich snippet generation with highlighted matching terms
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.inverted_index import InvertedIndex, Posting
from src.normalizer import TextNormalizer
from src.tokenizer import Tokenizer
from src.wildcard_processor import WildcardProcessor


@dataclass
class SearchResult:
    """
    Represents a scored matching document returned for a search query.

    Attributes:
        doc_id: Unique document identifier.
        title: Title of the document.
        score: Relevance score (1.0 for exact boolean match, TF-IDF score for ranked).
        matched_terms: List of query terms found in this document.
        snippet: Excerpt showing matched terms in document context.
        file_path: File system path if loaded from disk.
    """
    doc_id: int
    title: str
    score: float
    matched_terms: List[str] = field(default_factory=list)
    snippet: str = ""
    file_path: Optional[str] = None

    def __repr__(self) -> str:
        return f"SearchResult(doc_id={self.doc_id}, score={self.score:.4f}, title='{self.title}')"


class QueryProcessor:
    """
    Processes Boolean, Phrasal, Wildcard, and Ranked TF-IDF search queries.
    """

    def __init__(
        self,
        index: InvertedIndex,
        tokenizer: Optional[Tokenizer] = None,
        normalizer: Optional[TextNormalizer] = None,
        wildcard_processor: Optional[WildcardProcessor] = None,
    ):
        self.index = index
        self.tokenizer = tokenizer or Tokenizer()
        self.normalizer = normalizer or TextNormalizer()
        self.wildcard_processor = wildcard_processor or WildcardProcessor()

        # Keep wildcard processor synchronized with index vocabulary
        self.sync_wildcard_index()

    def sync_wildcard_index(self) -> None:
        """Synchronizes the wildcard processor with current index vocabulary."""
        vocab = self.index.get_vocabulary()
        self.wildcard_processor.build(vocab)

    # =========================================================================
    # Postings List Boolean Operations
    # =========================================================================

    @staticmethod
    def intersect_postings(list1: List[Posting], list2: List[Posting]) -> List[Posting]:
        """
        Boolean AND: Intersects two postings lists sorted by doc_id in O(len1 + len2).
        """
        result: List[Posting] = []
        i, j = 0, 0
        while i < len(list1) and j < len(list2):
            if list1[i].doc_id == list2[j].doc_id:
                # Merge positions if needed
                merged = Posting(
                    doc_id=list1[i].doc_id,
                    tf=list1[i].tf + list2[j].tf,
                    positions=sorted(list1[i].positions + list2[j].positions),
                )
                result.append(merged)
                i += 1
                j += 1
            elif list1[i].doc_id < list2[j].doc_id:
                i += 1
            else:
                j += 1
        return result

    @staticmethod
    def union_postings(list1: List[Posting], list2: List[Posting]) -> List[Posting]:
        """
        Boolean OR: Computes union of two sorted postings lists in O(len1 + len2).
        """
        result: List[Posting] = []
        i, j = 0, 0
        while i < len(list1) and j < len(list2):
            if list1[i].doc_id == list2[j].doc_id:
                merged = Posting(
                    doc_id=list1[i].doc_id,
                    tf=list1[i].tf + list2[j].tf,
                    positions=sorted(list1[i].positions + list2[j].positions),
                )
                result.append(merged)
                i += 1
                j += 1
            elif list1[i].doc_id < list2[j].doc_id:
                result.append(list1[i])
                i += 1
            else:
                result.append(list2[j])
                j += 1

        while i < len(list1):
            result.append(list1[i])
            i += 1
        while j < len(list2):
            result.append(list2[j])
            j += 1
        return result

    def difference_postings(self, list1: List[Posting], list2: List[Posting]) -> List[Posting]:
        """
        Boolean NOT / AND NOT: Retains postings in list1 whose doc_id is NOT in list2.
        """
        result: List[Posting] = []
        exclude_ids = {p.doc_id for p in list2}
        for p in list1:
            if p.doc_id not in exclude_ids:
                result.append(p)
        return result

    # =========================================================================
    # Phrase Query Processing
    # =========================================================================

    def search_phrase(self, phrase: str) -> List[SearchResult]:
        """
        Evaluates an exact phrase query where words must appear contiguously in order.
        """
        raw_tokens = self.tokenizer.tokenize(phrase)
        norm_tokens = self.normalizer.normalize_tokens(raw_tokens)

        if not norm_tokens:
            return []

        if len(norm_tokens) == 1:
            term = norm_tokens[0]
            postings = self.index.get_postings(term)
            return [
                self._build_search_result(p.doc_id, score=1.0, matched_terms=[term])
                for p in postings
            ]

        # Multi-word phrase positional intersection
        current_postings = self.index.get_postings(norm_tokens[0])

        for next_term in norm_tokens[1:]:
            next_postings = self.index.get_postings(next_term)
            if not current_postings or not next_postings:
                return []

            matched_phrase_postings: List[Posting] = []
            i, j = 0, 0
            while i < len(current_postings) and j < len(next_postings):
                p1, p2 = current_postings[i], next_postings[j]
                if p1.doc_id == p2.doc_id:
                    # Check positional adjacency: pos in p2 == pos in p1 + 1
                    valid_positions: List[int] = []
                    pos1_set = set(p1.positions)
                    for pos2 in p2.positions:
                        if (pos2 - 1) in pos1_set:
                            valid_positions.append(pos2)

                    if valid_positions:
                        matched_phrase_postings.append(
                            Posting(
                                doc_id=p1.doc_id,
                                tf=len(valid_positions),
                                positions=valid_positions,
                            )
                        )
                    i += 1
                    j += 1
                elif p1.doc_id < p2.doc_id:
                    i += 1
                else:
                    j += 1

            current_postings = matched_phrase_postings

        results: List[SearchResult] = []
        for p in current_postings:
            res = self._build_search_result(
                p.doc_id, score=float(p.tf), matched_terms=norm_tokens
            )
            results.append(res)
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # =========================================================================
    # Wildcard Expansion
    # =========================================================================

    def expand_wildcard_term(self, wildcard_token: str) -> Tuple[List[str], List[Posting]]:
        """
        Resolves a wildcard token into matching vocabulary terms and their union postings.
        """
        matching_terms = self.wildcard_processor.resolve(wildcard_token)
        if not matching_terms:
            return [], []

        postings: List[Posting] = []
        for term in matching_terms:
            t_postings = self.index.get_postings(term)
            postings = self.union_postings(postings, t_postings)

        return matching_terms, postings

    # =========================================================================
    # Boolean Query Parsing & Evaluation
    # =========================================================================

    def search_boolean(self, query: str) -> List[SearchResult]:
        """
        Evaluates a Boolean query supporting AND, OR, NOT operators and wildcards.
        Operator precedence: NOT > AND > OR. Parentheses supported.
        """
        query_str = query.strip()
        if not query_str:
            return []

        # Tokenize query preserving AND, OR, NOT, parentheses, phrases, and words
        token_pattern = re.compile(r'\(|\)|\bAND\b|\bOR\b|\bNOT\b|"[^"]+"|[^\s()]+', re.IGNORECASE)
        raw_tokens = token_pattern.findall(query_str)
        if not raw_tokens:
            return []

        # Insert implicit AND where appropriate (e.g., 'term1 NOT term2' -> 'term1 AND NOT term2')
        def is_operand_or_close(t: str) -> bool:
            return t != "(" and t.upper() not in ("AND", "OR", "NOT")

        def is_operand_or_open_or_not(t: str) -> bool:
            return t != ")" and t.upper() not in ("AND", "OR")

        tokens: List[str] = []
        for idx, t in enumerate(raw_tokens):
            tokens.append(t)
            if idx + 1 < len(raw_tokens):
                nxt = raw_tokens[idx + 1]
                if is_operand_or_close(t) and is_operand_or_open_or_not(nxt):
                    tokens.append("AND")

        # Convert infix notation to postfix (Reverse Polish Notation) using Shunting-Yard
        precedence = {"NOT": 3, "AND": 2, "OR": 1}

        output_queue: List[str] = []
        op_stack: List[str] = []

        for token in tokens:
            upper = token.upper()
            if upper in ("AND", "OR", "NOT"):
                while (
                    op_stack
                    and op_stack[-1] != "("
                    and precedence.get(op_stack[-1], 0) >= precedence[upper]
                ):
                    output_queue.append(op_stack.pop())
                op_stack.append(upper)
            elif token == "(":
                op_stack.append("(")
            elif token == ")":
                while op_stack and op_stack[-1] != "(":
                    output_queue.append(op_stack.pop())
                if op_stack and op_stack[-1] == "(":
                    op_stack.pop()
            else:
                # Operand (word, wildcard, or quoted phrase)
                output_queue.append(token)

        while op_stack:
            output_queue.append(op_stack.pop())

        # Evaluate RPN stack
        eval_stack: List[Tuple[List[Posting], List[str]]] = []
        all_doc_postings = [
            Posting(doc_id=doc_id, tf=1, positions=[])
            for doc_id in self.index.get_all_doc_ids()
        ]

        for token in output_queue:
            upper = token.upper()
            if upper == "NOT":
                if not eval_stack:
                    continue
                postings, terms = eval_stack.pop()
                diff = self.difference_postings(all_doc_postings, postings)
                eval_stack.append((diff, [f"NOT({','.join(terms)})"]))
            elif upper in ("AND", "OR"):
                if len(eval_stack) < 2:
                    continue
                p2, t2 = eval_stack.pop()
                p1, t1 = eval_stack.pop()
                if upper == "AND":
                    merged = self.intersect_postings(p1, p2)
                else:
                    merged = self.union_postings(p1, p2)
                eval_stack.append((merged, t1 + t2))
            else:
                # Operand evaluation
                if token.startswith('"') and token.endswith('"'):
                    phrase_text = token[1:-1]
                    res = self.search_phrase(phrase_text)
                    term_postings = [
                        Posting(doc_id=r.doc_id, tf=int(r.score), positions=[])
                        for r in res
                    ]
                    eval_stack.append((term_postings, [phrase_text]))
                elif self.wildcard_processor.has_wildcard(token):
                    matched_terms, wildcard_postings = self.expand_wildcard_term(token)
                    eval_stack.append((wildcard_postings, matched_terms))
                else:
                    norm = self.normalizer.normalize(token)
                    if norm:
                        term_postings = self.index.get_postings(norm)
                        eval_stack.append((term_postings, [norm]))
                    else:
                        eval_stack.append(([], []))

        if not eval_stack:
            return []

        final_postings, matched_terms = eval_stack[-1]
        results: List[SearchResult] = []
        for p in final_postings:
            res = self._build_search_result(
                p.doc_id, score=1.0, matched_terms=matched_terms
            )
            results.append(res)
        return results

    # =========================================================================
    # Ranked Retrieval (TF-IDF & Cosine Similarity)
    # =========================================================================

    def search_ranked(self, query: str) -> List[SearchResult]:
        """
        Ranked free-text query using Vector Space Model with TF-IDF weighting
        and Cosine length normalization.
        """
        raw_tokens = self.tokenizer.tokenize(query)
        if not raw_tokens:
            return []

        expanded_terms: List[str] = []
        for tok in raw_tokens:
            if self.wildcard_processor.has_wildcard(tok):
                m_terms = self.wildcard_processor.resolve(tok)
                expanded_terms.extend(m_terms)
            else:
                norm = self.normalizer.normalize(tok)
                if norm:
                    expanded_terms.append(norm)

        if not expanded_terms:
            return []

        N = self.index.total_documents
        if N == 0:
            return []

        # Calculate query term weights: w_{t,q} = tf * idf
        query_tf: Dict[str, int] = {}
        for t in expanded_terms:
            query_tf[t] = query_tf.get(t, 0) + 1

        query_weights: Dict[str, float] = {}
        query_norm_sq = 0.0

        for t, tf in query_tf.items():
            df = self.index.get_doc_frequency(t)
            if df == 0:
                continue
            # Smoothed IDF: log10(1 + N / df)
            idf = math.log10(1.0 + (N / df))
            weight = (1.0 + math.log10(tf)) * idf
            query_weights[t] = weight
            query_norm_sq += weight * weight

        query_magnitude = math.sqrt(query_norm_sq) if query_norm_sq > 0 else 1.0

        # Accumulate document scores
        doc_scores: Dict[int, float] = {}
        doc_matched_terms: Dict[int, Set[str]] = {}

        for term, q_weight in query_weights.items():
            postings = self.index.get_postings(term)
            df = len(postings)
            idf = math.log10(1.0 + (N / df))

            for p in postings:
                d_id = p.doc_id
                doc_tf_weight = 1.0 + math.log10(p.tf) if p.tf > 0 else 0.0
                doc_term_weight = doc_tf_weight * idf

                dot_product = q_weight * doc_term_weight
                doc_scores[d_id] = doc_scores.get(d_id, 0.0) + dot_product

                if d_id not in doc_matched_terms:
                    doc_matched_terms[d_id] = set()
                doc_matched_terms[d_id].add(term)

        # Cosine length normalization
        results: List[SearchResult] = []
        for doc_id, dot_score in doc_scores.items():
            doc_len = self.index.doc_lengths.get(doc_id, 1)
            # Use document length as cosine normalization factor
            normalizer_factor = math.sqrt(doc_len) * query_magnitude
            final_score = dot_score / (normalizer_factor if normalizer_factor > 0 else 1.0)

            res = self._build_search_result(
                doc_id=doc_id,
                score=final_score,
                matched_terms=sorted(doc_matched_terms.get(doc_id, set())),
            )
            results.append(res)

        # Sort descending by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # =========================================================================
    # Unified Query Router
    # =========================================================================

    def search(self, query: str, mode: str = "auto") -> Tuple[List[SearchResult], str]:
        """
        Dispatches search query according to specified mode or auto-detection.

        Args:
            query: Raw query text.
            mode: 'auto', 'ranked', 'boolean', 'phrase', or 'wildcard'.

        Returns:
            Tuple of (search_results, resolved_mode_description).
        """
        clean_q = query.strip()
        if not clean_q:
            return [], "Empty query"

        mode = mode.lower()

        if mode == "phrase" or (mode == "auto" and clean_q.startswith('"') and clean_q.endswith('"')):
            phrase_content = clean_q.strip('"')
            results = self.search_phrase(phrase_content)
            return results, f"Phrase Query: \"{phrase_content}\""

        if mode == "boolean" or (
            mode == "auto" and any(op in clean_q for op in (" AND ", " OR ", " NOT ", "(", ")"))
        ):
            results = self.search_boolean(clean_q)
            return results, f"Boolean Query: {clean_q}"

        if mode == "wildcard" or (mode == "auto" and self.wildcard_processor.has_wildcard(clean_q)):
            # If it's a standalone wildcard query pattern
            words = clean_q.split()
            if len(words) == 1 and self.wildcard_processor.has_wildcard(words[0]):
                matched_terms, postings = self.expand_wildcard_term(words[0])
                results = [
                    self._build_search_result(p.doc_id, score=float(p.tf), matched_terms=matched_terms)
                    for p in postings
                ]
                results.sort(key=lambda r: r.score, reverse=True)
                desc = f"Wildcard Pattern '{clean_q}' expanded to: {matched_terms}"
                return results, desc

        # Default fallback: Ranked retrieval (TF-IDF + Cosine similarity)
        results = self.search_ranked(clean_q)
        return results, f"Ranked TF-IDF Query: {clean_q}"

    # =========================================================================
    # Helpers
    # =========================================================================

    def _build_search_result(
        self, doc_id: int, score: float, matched_terms: List[str]
    ) -> SearchResult:
        doc = self.index.get_document(doc_id)
        title = doc.title if doc else f"Document {doc_id}"
        file_path = doc.file_path if doc else None
        snippet = self._generate_snippet(doc.content if doc else "", matched_terms)

        return SearchResult(
            doc_id=doc_id,
            title=title,
            score=score,
            matched_terms=list(matched_terms),
            snippet=snippet,
            file_path=file_path,
        )

    def _generate_snippet(self, content: str, terms: List[str], window_chars: int = 160) -> str:
        """Extracts text excerpt around the first appearance of matched terms."""
        if not content:
            return ""

        lower_content = content.lower()
        earliest_pos = len(content)

        for term in terms:
            clean_term = term.strip("*? ").lower()
            if not clean_term:
                continue
            pos = lower_content.find(clean_term)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos

        if earliest_pos == len(content):
            # No term found in raw content, return head
            return " ".join(content.split()[:30]) + "..."

        start = max(0, earliest_pos - 40)
        end = min(len(content), start + window_chars)
        snippet = content[start:end].strip()

        prefix = "... " if start > 0 else ""
        suffix = " ..." if end < len(content) else ""
        return f"{prefix}{snippet}{suffix}"
