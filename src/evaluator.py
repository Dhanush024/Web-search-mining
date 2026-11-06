"""
Module: Evaluation Metrics (Information Retrieval Evaluation)
=============================================================
Calculates standard IR evaluation metrics for search engine benchmarking:
- Precision@K
- Recall@K
- Average Precision (AP)
- Mean Average Precision (MAP)
- F1-Score
- Mean Reciprocal Rank (MRR)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set


class Evaluator:
    """
    Computes Information Retrieval evaluation metrics given retrieved doc IDs
    and ground-truth relevant doc IDs.
    """

    @staticmethod
    def precision_at_k(retrieved: Sequence[int], relevant: Set[int], k: int) -> float:
        """
        Calculates Precision@K: proportion of top-k retrieved docs that are relevant.
        """
        if k <= 0:
            return 0.0
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant)
        return relevant_retrieved / k

    @staticmethod
    def recall_at_k(retrieved: Sequence[int], relevant: Set[int], k: int) -> float:
        """
        Calculates Recall@K: proportion of all relevant docs present in top-k retrieved.
        """
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant)
        return relevant_retrieved / len(relevant)

    @classmethod
    def f1_score_at_k(cls, retrieved: Sequence[int], relevant: Set[int], k: int) -> float:
        """Calculates F1-measure at rank k (harmonic mean of precision and recall)."""
        p = cls.precision_at_k(retrieved, relevant, k)
        r = cls.recall_at_k(retrieved, relevant, k)
        if (p + r) == 0.0:
            return 0.0
        return (2.0 * p * r) / (p + r)

    @staticmethod
    def average_precision(retrieved: Sequence[int], relevant: Set[int]) -> float:
        """
        Calculates Average Precision (AP) for a single query:
        AP = sum_{k=1}^N (P@k * rel(k)) / |relevant|
        """
        if not relevant or not retrieved:
            return 0.0

        hits = 0
        sum_precisions = 0.0

        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                hits += 1
                precision_at_rank = hits / rank
                sum_precisions += precision_at_rank

        return sum_precisions / len(relevant)

    @classmethod
    def mean_average_precision(
        cls, query_results: Sequence[Sequence[int]], query_ground_truths: Sequence[Set[int]]
    ) -> float:
        """
        Calculates Mean Average Precision (MAP) across multiple queries.
        """
        if not query_results or len(query_results) != len(query_ground_truths):
            return 0.0

        total_ap = sum(
            cls.average_precision(retrieved, relevant)
            for retrieved, relevant in zip(query_results, query_ground_truths)
        )
        return total_ap / len(query_results)

    @staticmethod
    def reciprocal_rank(retrieved: Sequence[int], relevant: Set[int]) -> float:
        """Calculates Reciprocal Rank (RR): 1 / rank of first relevant document."""
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0

    @classmethod
    def mean_reciprocal_rank(
        cls, query_results: Sequence[Sequence[int]], query_ground_truths: Sequence[Set[int]]
    ) -> float:
        """Calculates Mean Reciprocal Rank (MRR) across multiple queries."""
        if not query_results:
            return 0.0
        total_rr = sum(
            cls.reciprocal_rank(retrieved, relevant)
            for retrieved, relevant in zip(query_results, query_ground_truths)
        )
        return total_rr / len(query_results)

    @classmethod
    def evaluate_query(cls, retrieved: Sequence[int], relevant: Set[int], k: int = 5) -> Dict[str, float]:
        """Runs a complete evaluation report for a single query."""
        return {
            f"Precision@{k}": round(cls.precision_at_k(retrieved, relevant, k), 4),
            f"Recall@{k}": round(cls.recall_at_k(retrieved, relevant, k), 4),
            f"F1@{k}": round(cls.f1_score_at_k(retrieved, relevant, k), 4),
            "Average_Precision": round(cls.average_precision(retrieved, relevant), 4),
            "Reciprocal_Rank": round(cls.reciprocal_rank(retrieved, relevant), 4),
        }

