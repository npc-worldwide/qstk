#!/usr/bin/env python3
"""Demo script for CHSH text contextuality analysis using anaphoric resolution.

Implements the methodology from Lo et al. (2024) to measure contextuality
in language model predictions for anaphoric resolution tasks.

Usage:
    python text_contextuality_demo.py --model gpt2 --n-instances 100
"""

import argparse
import json
import sys
from typing import Dict, List

try:
    from qstk.text_contextuality import (
        AnaphoricSchemaExtractor,
        TextContextualityAnalyzer,
        CorpusContextualityDataset,
    )
    from qstk.chsh import calculate_s_value, check_violation
except ImportError:
    sys.path.insert(0, "../src")
    from qstk.text_contextuality import (
        AnaphoricSchemaExtractor,
        TextContextualityAnalyzer,
        CorpusContextualityDataset,
    )
    from qstk.chsh import calculate_s_value, check_violation


def mock_llm_predict(context: str, candidates: List[str]) -> Dict[str, float]:
    """Mock LLM prediction function for testing.
    
    In production, this would call a real language model.
    """
    import random
    random.seed(hash(context) % 2**32)
    
    # Simulate context-dependent bias
    if "tall" in context.lower() and context.count("tall") > 1:
        # Bias toward noun A when "tall" is query adjective
        return {c: random.gauss(0.7 if i < 3 else 0.3, 0.1) 
                for i, c in enumerate(candidates)}
    elif "short" in context.lower() and context.count("short") > 1:
        # Bias toward noun B when "short" is query adjective
        return {c: random.gauss(0.3 if i < 3 else 0.7, 0.1) 
                for i, c in enumerate(candidates)}
    else:
        return {c: random.gauss(0.5, 0.15) for c in candidates}


def run_synthetic_demo(n_instances: int = 100):
    """Run demo with synthetic/mock LLM predictions."""
    print("=" * 60)
    print("CHSH Text Contextuality Analyzer Demo")
    print("=" * 60)
    print()
    
    # Step 1: Generate anaphoric instances
    print(f"[1/4] Generating {n_instances} anaphoric resolution instances...")
    extractor = AnaphoricSchemaExtractor()
    instances = extractor.generate_instances(n_instances)
    print(f"  Generated {len(instances)} instances")
    print(f"  Sample: \"{instances[0].context}\"")
    print()
    
    # Step 2: Initialize analyzer
    print("[2/4] Initializing TextContextualityAnalyzer...")
    analyzer = TextContextualityAnalyzer(temperature=1.0)
    print("  Analyzer ready")
    print()
    
    # Step 3: Compute probabilities
    print("[3/4] Computing resolution probabilities (mock LLM)...")
    for instance in instances:
        analyzer.compute_resolution_probabilities(instance, mock_llm_predict)
    print(f"  Computed probabilities for {len(instances)} instances")
    print(f"  Sample P(A)={instances[0].probability_a:.3f}, P(B)={instances[0].probability_b:.3f}")
    print()
    
    # Step 4: Calculate CHSH S-value
    print("[4/4] Computing CHSH S-value...")
    results = analyzer.compute_contextuality(instances)
    print()
    print("-" * 60)
    print("RESULTS:")
    print("-" * 60)
    print(f"  S-value:              {results['s_value']:.4f}")
    print(f"  |S|:                  {results['abs_s']:.4f}")
    print(f"  Classical bound:      2.0")
    print(f"  Quantum bound:        {2 * (2 ** 0.5):.4f}")
    print(f"  Classical violation:  {results['classical_violation']}")
    print(f"  Quantum violation:    {results['quantum_violation']}")
    print(f"  N instances:          {results['n_instances']}")
    print()
    print("Expectation values:")
    for key, val in results.get('expectation_values', {}).items():
        print(f"    E({key}) = {val:.4f}")
    print()
    
    # Interpretation
    if results['classical_violation']:
        print("✓ CONTEXTUALITY DETECTED: S-value exceeds classical bound!")
        print("  This indicates non-classical correlations in the LLM predictions.")
    else:
        print("✗ No contextuality detected: S-value within classical bounds.")
    
    return results


def run_wikitext_demo(split: str = "train", max_sentences: int = 1000):
    """Run demo extracting patterns from WikiText corpus."""
    print("=" * 60)
    print("CHSH WikiText Contextuality Analysis")
    print("=" * 60)
    print()
    
    # Load dataset
    print(f"[1/3] Loading WikiText ({split})...")
    dataset = CorpusContextualityDataset(min_sentence_length=30, max_sentence_length=150)
    
    try:
        sentences = dataset.load_wikitext(split)
        print(f"  Loaded {len(sentences)} sentences")
    except Exception as e:
        print(f"  Error loading WikiText: {e}")
        print("  Install datasets: pip install datasets")
        return None
    
    # Filter for anaphoric patterns
    print("[2/3] Filtering for anaphoric patterns...")
    filtered = dataset.filter_anaphoric_sentences(sentences[:max_sentences])
    print(f"  Found {len(filtered)} potential anaphoric sentences")
    
    if len(filtered) < 4:
        print("  Not enough anaphoric sentences found. Using synthetic data instead.")
        return run_synthetic_demo(n_instances=100)
    
    # Extract instances and analyze
    print("[3/3] Extracting and analyzing instances...")
    extractor = AnaphoricSchemaExtractor()
    instances = []
    for sent in filtered[:100]:  # Limit to 100
        extracted = extractor.extract_from_text(sent)
        instances.extend(extracted)
    
    print(f"  Extracted {len(instances)} anaphoric instances")
    
    if len(instances) < 4:
        print("  Not enough instances extracted. Using synthetic data.")
        return run_synthetic_demo(n_instances=100)
    
    # Analyze
    analyzer = TextContextualityAnalyzer()
    for instance in instances:
        analyzer.compute_resolution_probabilities(instance, mock_llm_predict)
    
    results = analyzer.compute_contextuality(instances)
    
    print()
    print("-" * 60)
    print("RESULTS:")
    print("-" * 60)
    print(f"  S-value:              {results['s_value']:.4f}")
    print(f"  |S|:                  {results['abs_s']:.4f}")
    print(f"  Classical violation:  {results['classical_violation']}")
    print(f"  N instances:          {results['n_instances']}")
    print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="CHSH Text Contextuality Analyzer Demo"
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "wikitext"],
        default="synthetic",
        help="Demo mode: synthetic data or WikiText extraction"
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=100,
        help="Number of anaphoric instances to generate (synthetic mode)"
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=1000,
        help="Max sentences to scan from WikiText (wikitext mode)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    if args.mode == "synthetic":
        results = run_synthetic_demo(args.n_instances)
    else:
        results = run_wikitext_demo(max_sentences=args.max_sentences)
    
    if args.json and results:
        print("\nJSON output:")
        print(json.dumps({k: v for k, v in results.items() if k != "instances"}, indent=2))
    
    return 0 if results and results.get('s_value') is not None else 1


if __name__ == "__main__":
    sys.exit(main())
