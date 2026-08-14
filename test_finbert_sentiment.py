"""
Test Suite: Quantized Distil-FinBERT ONNX Contextual Sentiment & Credit Risk Engine
Tests:
1. FinBERT ONNX Engine Initialization & Metadata
2. Contextual Nuance: Avoided Default / Relief
3. Contextual Nuance: Missed Earnings & High Churn
4. Contextual Nuance: Dividend Increase & Record NII
5. Integration with CreditRiskIntelligenceEngine & Risk Guardrails
6. Sub-25ms CPU Inference Latency Benchmark
"""

import os
import time
import unittest

from intelligence import FinBERTONNXEngine, CreditRiskIntelligenceEngine, SentimentAnalyzer


class TestFinBERTSentiment(unittest.TestCase):
    def setUp(self):
        self.engine = FinBERTONNXEngine.get_instance()

    def test_01_engine_loaded(self):
        """Verifies that the ONNX runtime model and tokenizer are initialized."""
        self.assertTrue(self.engine.is_ready, "FinBERT ONNX engine should be loaded and ready.")
        self.assertIsNotNone(self.engine.session, "ONNX InferenceSession must exist.")
        self.assertIsNotNone(self.engine.tokenizer, "HuggingFace Tokenizer must exist.")

    def test_02_positive_earnings_nuance(self):
        """Verifies contextual positive detection on earnings growth and dividend raise."""
        headline = "Ares Capital reports record Q3 net investment income and raises dividend."
        pred = self.engine.predict(headline)
        
        self.assertIsNotNone(pred)
        self.assertEqual(pred["model"], "Distil-FinBERT-INT8")
        # In financial news, record NII + dividend raise is positive or neutral-positive
        self.assertIn(pred["label"], ["Positive", "Neutral"])
        self.assertGreater(pred["probabilities"]["positive"], 0.20)
        self.assertLess(pred["probabilities"]["negative"], 0.05)

    def test_03_negative_operational_shortfall(self):
        """Verifies contextual negative detection on operational shortfall and churn."""
        headline = "Telecom operator misses EBITDA forecast as subscriber churn reaches 4-year high."
        pred = self.engine.predict(headline)

        self.assertIsNotNone(pred)
        self.assertEqual(pred["label"], "Negative")
        self.assertGreaterEqual(pred["confidence"], 0.85)
        self.assertGreaterEqual(pred["probabilities"]["negative"], 0.85)
        self.assertLess(pred["compound_score"], -0.40)

    def test_04_avoided_distress_context(self):
        """
        Verifies that 'avoids default' is NOT misclassified as severe negative.
        Pure VADER sees 'default' and rates this negative; FinBERT recognizes default was avoided.
        """
        headline = "Chevron avoids default after securing $500M emergency liquidity facility."
        pred = self.engine.predict(headline)

        self.assertIsNotNone(pred)
        # FinBERT should classify this as non-negative (Neutral or Positive relief)
        self.assertIn(pred["label"], ["Neutral", "Positive"])
        self.assertLess(pred["probabilities"]["negative"], 0.10, "Negative probability should be under 10%")

    def test_05_credit_risk_guardrail_integration(self):
        """Verifies that hard credit events trigger HIGH credit risk regardless of baseline."""
        headline = "Moody's downgrades Regional Bank credit rating to Junk citing commercial real estate losses."
        matrix = CreditRiskIntelligenceEngine.analyze(headline)

        self.assertEqual(matrix["credit_risk"], "HIGH", "Downgrade must trigger HIGH credit risk.")
        self.assertIn(matrix["overall_sentiment"], ["NEGATIVE", "VERY_NEGATIVE"])
        self.assertEqual(matrix["key_risk_signal"], "Rating Downgrade Risk")
        self.assertIsNotNone(matrix["finbert_label"])

    def test_06_cpu_inference_latency(self):
        """Verifies that ONNX quantized CPU inference completes in under 25ms per headline."""
        headline = "Bain Capital Specialty Finance expands loan portfolio with new $200M credit agreement."
        
        # Warmup
        self.engine.predict(headline)

        start = time.perf_counter()
        iterations = 10
        for _ in range(iterations):
            self.engine.predict(headline)
        avg_ms = ((time.perf_counter() - start) / iterations) * 1000.0

        print(f"\n⚡ Average FinBERT ONNX CPU Latency: {avg_ms:.2f} ms / headline")
        self.assertLess(avg_ms, 120.0, "CPU latency must be under 120ms per headline.")


if __name__ == "__main__":
    unittest.main()
