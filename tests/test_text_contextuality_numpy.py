#!/usr/bin/env python3
"""NumPy-only implementation of CHSH text contextuality analyzer.

This is a standalone version that works without PyTorch or transformers,
suitable for testing the CHSH computation pipeline in resource-constrained
environments.

Example usage:
    python test_text_contextuality_numpy.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random


# === CHSH Functions (from qstk.chsh.py) ===

def calculate_s_value(expectation_values: Dict[str, float]) -> float:
    """Calculate CHSH S-value from expectation values."""
    s = (
        expectation_values.get("A", 0.0) +
        expectation_values.get("A_prime", 0.0) +
        expectation_values.get("B", 0.0) -
        expectation_values.get("B_prime", 0.0)
    )
    return s


def check_violation(s_value: float, bound: float = 2.0) -> bool:
    """Check if S-value violates CHSH inequality."""
    return abs(s_value) > bound


def compute_chsh_products(outcomes: Dict[str, List[float]]) -> Dict[str, float]:
    """Compute CHSH products from outcomes."""
    products = {}
    
    for key in ["A", "A_prime", "B", "B_prime"]:
        if key in outcomes and outcomes[key]:
            products[key] = np.mean(outcomes[key])
        else:
            products[key] = 0.0
    
    return products


# === Text Contextuality Classes ===

@dataclass
class AnaphoricInstance:
    """Single anaphoric resolution instance following Lo et al. schema."""
    context: str
    adj_a: str
    adj_b: str
    noun_a: str
    noun_b: str
    query_adjective: str
    correct_resolution: str
    probability_a: Optional[float] = None
    probability_b: Optional[float] = None


class AnaphoricSchemaExtractor:
    """Extract anaphoric resolution instances following Lo et al. (2024)."""
    
    ANAPHORIC_TEMPLATE = "The {adj_a} {noun_a} and the {adj_b} {noun_b} are related. The {query_adj} one is"
    
    ADJECTIVE_PAIRS = [
        ("tall", "short"), ("big", "small"), ("old", "young"),
        ("fast", "slow"), ("smart", "dull"), ("rich", "poor"),
        ("loud", "quiet"), ("heavy", "light"), ("strong", "weak"),
    ]
    
    NOUN_PAIRS = [
        ("man", "boy"), ("woman", "girl"), ("teacher", "student"),
        ("cat", "mouse"), ("lion", "cub"), ("tree", "flower"),
    ]
    
    def __init__(self):
        self.instances: List[AnaphoricInstance] = []
    
    def generate_instances(self, n_instances: int = 100) -> List[AnaphoricInstance]:
        """Generate anaphoric instances from schema."""
        instances = []
        random.seed(42)
        
        while len(instances) < n_instances:
            for adj_pair in self.ADJECTIVE_PAIRS:
                for noun_pair in self.NOUN_PAIRS:
                    if len(instances) >= n_instances:
                        break
                    
                    for query_idx in [0, 1]:
                        if len(instances) >= n_instances:
                            break
                            
                        context = self.ANAPHORIC_TEMPLATE.format(
                            adj_a=adj_pair[0],
                            noun_a=noun_pair[0],
                            adj_b=adj_pair[1],
                            noun_b=noun_pair[1],
                            query_adj=adj_pair[query_idx]
                        )
                        
                        correct_resolution = "A" if query_idx == 0 else "B"
                        
                        instances.append(AnaphoricInstance(
                            context=context,
                            adj_a=adj_pair[0],
                            adj_b=adj_pair[1],
                            noun_a=noun_pair[0],
                            noun_b=noun_pair[1],
                            query_adjective=adj_pair[query_idx],
                            correct_resolution=correct_resolution
                        ))
        
        self.instances = instances
        return instances


class TextContextualityAnalyzer:
    """Analyze contextuality in text using anaphoric resolution probabilities."""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def compute_resolution_probabilities(
        self, 
        instance: AnaphoricInstance, 
        llm_predict_fn=None
    ) -> Dict[str, float]:
        """Compute P(pronoun | context)."""
        if llm_predict_fn is None:
            return {"A": 0.5, "B": 0.5}
        
        candidates = ["he", "him", "she", "her", "they", "them"]
        logits = llm_predict_fn(instance.context, candidates)
        
        # Manual softmax
        import math
        exp_logits = [math.exp(logits.get(c, 0.0) / self.temperature) for c in candidates]
        total = sum(exp_logits)
        probabilities = {c: exp_logits[i] / total for i, c in enumerate(candidates)}
        
        # Aggregate to P(A) vs P(B)
        p_a = sum(probabilities.get(tok, 0) for tok in ["he", "him", "his"])
        p_b = sum(probabilities.get(tok, 0) for tok in ["she", "her", "hers"])
        
        total_ab = p_a + p_b
        if total_ab > 0:
            p_a /= total_ab
            p_b /= total_ab
        else:
            p_a = p_b = 0.5
        
        instance.probability_a = p_a
        instance.probability_b = p_b
        
        return {"A": p_a, "B": p_b}
    
    def construct_chsh_settings(
        self, 
        instances: List[AnaphoricInstance]
    ) -> Dict[str, List[float]]:
        """Construct CHSH measurement settings from anaphoric instances."""
        if not instances:
            return {"A": [], "A_prime": [], "B": [], "B_prime": []}
        
        outcomes = {"A": [], "A_prime": [], "B": [], "B_prime": []}
        
        for i, instance in enumerate(instances):
            setting_idx = i % 4
            
            if instance.probability_a is not None and instance.probability_b is not None:
                outcome = instance.probability_a - instance.probability_b
            else:
                outcome = 0.0
            
            if setting_idx == 0:
                outcomes["A"].append(outcome)
            elif setting_idx == 1:
                outcomes["A_prime"].append(outcome)
            elif setting_idx == 2:
                outcomes["B"].append(outcome)
            else:
                outcomes["B_prime"].append(outcome)
        
        return outcomes
    
    def compute_contextuality(self, instances: List[AnaphoricInstance]) -> Dict:
        """Compute CHSH S-value from anaphoric instances."""
        
        outcomes = self.construct_chsh_settings(instances)
        products = compute_chsh_products(outcomes)
        
        if not products:
            return {"error": "Could not compute CHSH products", "s_value": None}
        
        expectation_values = {k: v for k, v in products.items()}
        
        s_value = calculate_s_value(expectation_values)
        violates = check_violation(s_value, bound=2.0)
        violates_quantum = abs(s_value) > 2 * np.sqrt(2)
        
        return {
            "s_value": s_value,
            "abs_s": abs(s_value),
            "classical_violation": violates,
            "quantum_violation": violates_quantum,
            "expectation_values": expectation_values,
            "n_instances": len(instances)
        }
    
    def batch_analyze(
        self, 
        n_instances: int = 100, 
        llm_predict_fn=None
    ) -> Dict:
        """Full analysis pipeline."""
        extractor = AnaphoricSchemaExtractor()
        instances = extractor.generate_instances(n_instances)
        
        for instance in instances:
            self.compute_resolution_probabilities(instance, llm_predict_fn)
        
        results = self.compute_contextuality(instances)
        results["instances"] = instances
        return results


def mock_llm(context: str, candidates: List[str]) -> Dict[str, float]:
    """Mock LLM with context-dependent bias for testing."""
    random.seed(hash(context) % 2**32)
    
    if "tall" in context.lower() and context.count("tall") > 1:
        return {"he": 0.8, "him": 0.7, "his": 0.6, 
                "she": 0.2, "her": 0.2, "hers": 0.1}
    elif "short" in context.lower() and context.count("short") > 1:
        return {"he": 0.2, "him": 0.2, "his": 0.1,
                "she": 0.8, "her": 0.7, "hers": 0.6}
    else:
        return {c: 0.5 for c in candidates}


def run_test():
    """Run the CHSH text contextuality test."""
    print("=" * 70)
    print("CHSH Text Contextuality Analyzer - Pure NumPy Test")
    print("=" * 70)
    print()
    
    # Test 1: Generate anaphoric instances
    print("[1/4] Testing AnaphoricSchemaExtractor...")
    extractor = AnaphoricSchemaExtractor()
    instances = extractor.generate_instances(n_instances=32)
    print(f"  Generated {len(instances)} anaphoric instances")
    print(f"  Sample: \"{instances[0].context}\"")
    print(f"  Correct resolution: {instances[0].correct_resolution}")
    print()
    
    # Test 2: Compute probabilities
    print("[2/4] Testing probability computation...")
    analyzer = TextContextualityAnalyzer(temperature=1.0)
    
    for instance in instances:
        analyzer.compute_resolution_probabilities(instance, mock_llm)
    
    print(f"  Computed probabilities for {len(instances)} instances")
    print(f"  Sample P(A)={instances[0].probability_a:.3f}, P(B)={instances[0].probability_b:.3f}")
    print()
    
    # Test 3: Construct CHSH settings
    print("[3/4] Testing CHSH settings construction...")
    settings = analyzer.construct_chsh_settings(instances)
    print(f"  Setting A: {len(settings['A'])} outcomes")
    print(f"  Setting A': {len(settings['A_prime'])} outcomes")
    print(f"  Setting B: {len(settings['B'])} outcomes")
    print(f"  Setting B': {len(settings['B_prime'])} outcomes")
    
    all_outcomes = settings['A'] + settings['A_prime'] + settings['B'] + settings['B_prime']
    print(f"  Outcome range: [{min(all_outcomes):.3f}, {max(all_outcomes):.3f}]")
    print()
    
    # Test 4: Compute CHSH S-value
    print("[4/4] Computing CHSH S-value...")
    results = analyzer.compute_contextuality(instances)
    
    print()
    print("-" * 70)
    print("RESULTS:")
    print("-" * 70)
    print(f"  S-value:              {results['s_value']:.4f}")
    print(f"  |S|:                  {results['abs_s']:.4f}")
    print(f"  Classical bound:      2.0")
    print(f"  Quantum bound:        {2 * np.sqrt(2):.4f}")
    print(f"  Classical violation:  {results['classical_violation']}")
    print(f"  Quantum violation:    {results['quantum_violation']}")
    print()
    
    print("Expectation values:")
    for key, val in results.get('expectation_values', {}).items():
        print(f"    E({key}) = {val:.4f}")
    print()
    
    if results['classical_violation']:
        print("✓ CONTEXTUALITY DETECTED: S-value exceeds classical bound!")
    else:
        print("✗ No contextuality detected: S-value within classical bounds.")
    
    print()
    print("=" * 70)
    print("Test complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_test()
