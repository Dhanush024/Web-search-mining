"""
Unit tests for Information Retrieval Evaluation Metrics (Evaluator)
"""

import unittest
from src.evaluator import Evaluator


class TestEvaluator(unittest.TestCase):

    def test_precision_at_k(self):
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 7}
        # At k=3, top 3 are [1, 2, 3]. Relevant are {1, 3} -> 2/3
        self.assertAlmostEqual(Evaluator.precision_at_k(retrieved, relevant, k=3), 2 / 3)
        # At k=5, top 5 are [1, 2, 3, 4, 5]. Relevant are {1, 3} -> 2/5 = 0.4
        self.assertAlmostEqual(Evaluator.precision_at_k(retrieved, relevant, k=5), 0.4)

    def test_recall_at_k(self):
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 7}
        # Out of 3 total relevant docs {1, 3, 7}, 2 are in top 5 -> 2/3
        self.assertAlmostEqual(Evaluator.recall_at_k(retrieved, relevant, k=5), 2 / 3)

    def test_f1_score_at_k(self):
        retrieved = [1, 2]
        relevant = {1, 2}
        # Perfect precision (1.0) and recall (1.0) -> F1 = 1.0
        self.assertAlmostEqual(Evaluator.f1_score_at_k(retrieved, relevant, k=2), 1.0)

    def test_average_precision(self):
        # Retrieved: [1, 2, 3, 4, 5]
        # Relevant: {1, 3}
        # Rank 1: doc 1 is relevant -> P@1 = 1/1 = 1.0
        # Rank 2: doc 2 is not relevant
        # Rank 3: doc 3 is relevant -> P@3 = 2/3
        # AP = (1.0 + 2/3) / 2 = (5/3) / 2 = 5/6 = 0.8333...
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3}
        self.assertAlmostEqual(Evaluator.average_precision(retrieved, relevant), 5 / 6)

    def test_mean_average_precision(self):
        q1_retrieved = [1, 2, 3]
        q1_relevant = {1}  # AP = 1.0

        q2_retrieved = [2, 1, 3]
        q2_relevant = {1}  # Rank 2 -> P@2 = 1/2 = 0.5. AP = 0.5

        # MAP = (1.0 + 0.5) / 2 = 0.75
        map_score = Evaluator.mean_average_precision(
            [q1_retrieved, q2_retrieved],
            [q1_relevant, q2_relevant]
        )
        self.assertAlmostEqual(map_score, 0.75)

    def test_reciprocal_rank(self):
        retrieved = [5, 4, 3, 2, 1]
        relevant = {3, 1}
        # First relevant doc is at rank 3 -> RR = 1/3
        self.assertAlmostEqual(Evaluator.reciprocal_rank(retrieved, relevant), 1 / 3)

    def test_evaluate_query_summary(self):
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 2}
        report = Evaluator.evaluate_query(retrieved, relevant, k=2)
        self.assertEqual(report["Precision@2"], 1.0)
        self.assertEqual(report["Recall@2"], 1.0)
        self.assertEqual(report["F1@2"], 1.0)


if __name__ == "__main__":
    unittest.main()

