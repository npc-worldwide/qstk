"""Unit tests for text contextuality analyzer.

Tests the anaphoric resolution schema and CHSH computation pipeline.
"""

import sys
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from qstk.text_contextuality import (
    AnaphoricInstance,
    AnaphoricSchemaExtractor,
    TextContextualityAnalyzer,
    CorpusContextualityDataset,
)


class TestAnaphoricSchemaExtractor(unittest.TestCase):
    """Tests for AnaphoricSchemaExtractor."""
    
    def test_generate_instances(self):
        """Test instance generation."""
        extractor = AnaphoricSchemaExtractor()
        instances = extractor.generate_instances(n_instances=20)
        
        self.assertEqual(len(instances), 20)
        
        # Check instance structure
        for inst in instances:
            self.assertIsInstance(inst, AnaphoricInstance)
            self.assertIsNotNone(inst.context)
            self.assertIsNotNone(inst.adj_a)
            self.assertIsNotNone(inst.adj_b)
            self.assertIsNotNone(inst.noun_a)
            self.assertIsNotNone(inst.noun_b)
            self.assertIsNotNone(inst.query_adjective)
            self.assertIn(inst.correct_resolution, ["A", "B"])
    
    def test_instance_context_format(self):
        """Test that context follows expected template."""
        extractor = AnaphoricSchemaExtractor()
        instances = extractor.generate_instances(n_instances=4)
        
        # Check pattern: "The X Y and the Z W are related. The V one is"
        for inst in instances:
            self.assertIn("and the", inst.context.lower())
            self.assertIn("are related", inst.context.lower())
            self.assertIn("one is", inst.context.lower())
    
    def test_extract_from_text(self):
        """Test extraction from natural text."""
        extractor = AnaphoricSchemaExtractor()
        
        text = "The tall man and the short boy are related. The tall one is he."
        instances = extractor.extract_from_text(text)
        
        # Should find at least one match
        self.assertGreaterEqual(len(instances), 1)
        
        # Check extracted content
        inst = instances[0]
        self.assertEqual(inst.adj_a, "tall")
        self.assertEqual(inst.adj_b, "short")
        self.assertEqual(inst.noun_a, "man")
        self.assertEqual(inst.noun_b, "boy")


class TestTextContextualityAnalyzer(unittest.TestCase):
    """Tests for TextContextualityAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TextContextualityAnalyzer(temperature=1.0)
        self.extractor = AnaphoricSchemaExtractor()
        self.instances = self.extractor.generate_instances(n_instances=16)
    
    def test_compute_resolution_probabilities(self):
        """Test probability computation."""
        instance = self.instances[0]
        
        # Test with no LLM function (dummy mode)
        probs = self.analyzer.compute_resolution_probabilities(instance)
        
        self.assertIn("A", probs)
        self.assertIn("B", probs)
        self.assertAlmostEqual(probs["A"] + probs["B"], 1.0, places=5)
        
        # Instance should be updated
        self.assertIsNotNone(instance.probability_a)
        self.assertIsNotNone(instance.probability_b)
    
    def test_compute_resolution_probabilities_with_llm(self):
        """Test probability computation with mock LLM."""
        instance = self.instances[0]
        
        def mock_llm(context, candidates):
            # Return deterministic logits
            return {c: 1.0 if i == 0 else 0.5 for i, c in enumerate(candidates)}
        
        probs = self.analyzer.compute_resolution_probabilities(instance, mock_llm)
        
        self.assertIn("A", probs)
        self.assertIn("B", probs)
        self.assertGreater(probs["A"], 0)
        self.assertGreater(probs["B"], 0)
    
    def test_construct_chsh_settings(self):
        """Test CHSH settings construction."""
        # Compute probabilities first
        for inst in self.instances:
            self.analyzer.compute_resolution_probabilities(inst)
        
        settings = self.analyzer.construct_chsh_settings(self.instances)
        
        self.assertIn("A", settings)
        self.assertIn("A_prime", settings)
        self.assertIn("B", settings)
        self.assertIn("B_prime", settings)
        
        # Each setting should have 4 outcomes (16 instances / 4 settings)
        self.assertEqual(len(settings["A"]), 4)
        self.assertEqual(len(settings["A_prime"]), 4)
        self.assertEqual(len(settings["B"]), 4)
        self.assertEqual(len(settings["B_prime"]), 4)
        
        # Outcomes should be in [-1, 1]
        for setting_name, outcomes in settings.items():
            for outcome in outcomes:
                self.assertGreaterEqual(outcome, -1.0)
                self.assertLessEqual(outcome, 1.0)
    
    def test_compute_contextuality(self):
        """Test CHSH S-value computation."""
        # Compute probabilities
        for inst in self.instances:
            self.analyzer.compute_resolution_probabilities(inst)
        
        results = self.analyzer.compute_contextuality(self.instances)
        
        self.assertIn("s_value", results)
        self.assertIn("abs_s", results)
        self.assertIn("classical_violation", results)
        self.assertIn("quantum_violation", results)
        self.assertIn("expectation_values", results)
        
        # S-value should be a number
        self.assertIsInstance(results["s_value"], (int, float))
        
        # abs_s should be non-negative
        self.assertGreaterEqual(results["abs_s"], 0)
    
    def test_batch_analyze(self):
        """Test full pipeline."""
        results = self.analyzer.batch_analyze(n_instances=16)
        
        self.assertIn("s_value", results)
        self.assertIn("instances", results)
        self.assertEqual(len(results["instances"]), 16)


class TestCorpusContextualityDataset(unittest.TestCase):
    """Tests for CorpusContextualityDataset."""
    
    def test_filter_anaphoric_sentences(self):
        """Test anaphoric sentence filtering."""
        dataset = CorpusContextualityDataset()
        
        sentences = [
            "The tall man and the short boy are related.",
            "The cat sat on the mat.",
            "The smart student and the dull teacher are related. The smart one is he.",
            "Hello world.",
        ]
        
        filtered = dataset.filter_anaphoric_sentences(sentences)
        
        # Should find 2 matching sentences
        self.assertEqual(len(filtered), 2)
        self.assertIn(sentences[0], filtered)
        self.assertIn(sentences[2], filtered)
    
    def test_filter_empty_list(self):
        """Test filtering empty list."""
        dataset = CorpusContextualityDataset()
        filtered = dataset.filter_anaphoric_sentences([])
        self.assertEqual(len(filtered), 0)


class TestAnaphoricInstance(unittest.TestCase):
    """Tests for AnaphoricInstance dataclass."""
    
    def test_instance_creation(self):
        """Test instance creation."""
        instance = AnaphoricInstance(
            context="The tall man and the short boy are related. The tall one is",
            adj_a="tall",
            adj_b="short",
            noun_a="man",
            noun_b="boy",
            query_adjective="tall",
            correct_resolution="A",
            probability_a=0.7,
            probability_b=0.3,
        )
        
        self.assertEqual(instance.adj_a, "tall")
        self.assertEqual(instance.probability_a, 0.7)
        self.assertEqual(instance.correct_resolution, "A")
    
    def test_instance_defaults(self):
        """Test default values."""
        instance = AnaphoricInstance(
            context="Test",
            adj_a="big",
            adj_b="small",
            noun_a="cat",
            noun_b="dog",
            query_adjective="big",
            correct_resolution="A",
        )
        
        self.assertIsNone(instance.probability_a)
        self.assertIsNone(instance.probability_b)


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""
    
    def test_full_pipeline(self):
        """Test complete workflow from extraction to CHSH."""
        # Generate instances
        extractor = AnaphoricSchemaExtractor()
        instances = extractor.generate_instances(n_instances=32)
        
        # Analyze
        analyzer = TextContextualityAnalyzer()
        
        # Mock LLM that always prefers noun A
        def biased_llm(context, candidates):
            return {c: 0.8 if i < 3 else 0.2 for i, c in enumerate(candidates)}
        
        results = analyzer.batch_analyze(
            n_instances=32,
            llm_predict_fn=biased_llm
        )
        
        # Verify structure
        self.assertIn("s_value", results)
        self.assertIn("classical_violation", results)
        self.assertIsInstance(results["s_value"], float)
        
        # Check that probabilities were computed
        for inst in results["instances"]:
            self.assertIsNotNone(inst.probability_a)
            self.assertIsNotNone(inst.probability_b)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestAnaphoricSchemaExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestTextContextualityAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestCorpusContextualityDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestAnaphoricInstance))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
