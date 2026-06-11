"""Text-based contextuality analysis using anaphoric resolution schema.

Implements the linguistic contextuality measurement framework from:
- Lo, Sadrzadeh, Mansfield (2024): "Quantum-Like Contextuality in Large Language Models"
- arXiv:2412.16806

The anaphoric resolution schema extracts probability distributions from LLM
predictions on sentences where pronoun references are ambiguous based on
adjective modification (e.g., "the tall man and the short boy... the tall one is he")
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .chsh import calculate_s_value, check_violation, compute_chsh_products


@dataclass
class AnaphoricInstance:
    """Single anaphoric resolution instance following Lo et al. schema."""
    context: str  # Full sentence with masked pronoun
    adj_a: str  # Adjective modifying noun A
    adj_b: str  # Adjective modifying noun B
    noun_a: str  # First noun
    noun_b: str  # Second noun
    query_adjective: str  # Adjective that determines resolution
    correct_resolution: str  # Which noun ("A" or "B") the pronoun refers to
    probability_a: Optional[float] = None  # P(pronoun = noun_a)
    probability_b: Optional[float] = None  # P(pronoun = noun_b)


class AnaphoricSchemaExtractor:
    """Extract anaphoric resolution instances following Lo et al. (2024)."""
    
    # Template for anaphoric sentences
    ANAPHORIC_TEMPLATE = "The {adj_a} {noun_a} and the {adj_b} {noun_b} are related. The {query_adj} one is"
    
    # Sample adjective pairs (contrasting scales)
    ADJECTIVE_PAIRS = [
        ("tall", "short"), ("big", "small"), ("old", "young"),
        ("fast", "slow"), ("smart", "dull"), ("rich", "poor"),
        ("loud", "quiet"), ("heavy", "light"), ("strong", "weak"),
    ]
    
    # Sample noun pairs
    NOUN_PAIRS = [
        ("man", "boy"), ("woman", "girl"), ("teacher", "student"),
        ("cat", "mouse"), ("lion", "cub"), ("tree", "flower"),
    ]
    
    # Pronoun forms
    PRONOUNS_A = ["he", "him", "his", "this"]
    PRONOUNS_B = ["she", "her", "hers", "that"]
    PRONOUNS_NEUTRAL = ["they", "them"]
    
    def __init__(self):
        self.instances: List[AnaphoricInstance] = []
    
    def generate_instances(self, n_instances: int = 100) -> List[AnaphoricInstance]:
        """Generate anaphoric instances from schema."""
        instances = []
        np.random.seed(42)
        
        while len(instances) < n_instances:
            for adj_pair in self.ADJECTIVE_PAIRS:
                for noun_pair in self.NOUN_PAIRS:
                    if len(instances) >= n_instances:
                        break
                    
                    # Alternate which adjective is used for query
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
                        
                        # The pronoun resolves to the noun matching query_adj
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
    
    def extract_from_text(self, text: str) -> List[AnaphoricInstance]:
        """Extract anaphoric patterns from natural text using regex."""
        pattern = re.compile(
            r"the\s+(\w+)\s+(\w+)\s+and\s+the\s+(\w+)\s+(\w+)" + 
            r".*?the\s+(?:(first|second|same|\w+))\s+one\s+is\s+(\w+)",
            re.IGNORECASE
        )
        
        instances = []
        for match in pattern.finditer(text):
            groups = match.groups()
            if len(groups) >= 6:
                resolution = "A" if groups[5].lower() in self.PRONOUNS_A else "B"
                instance = AnaphoricInstance(
                    context=text[match.start():match.end()],
                    adj_a=groups[0],
                    adj_b=groups[2],
                    noun_a=groups[1],
                    noun_b=groups[3],
                    query_adjective=groups[4] if groups[4] in ["first", "second"] else groups[4],
                    correct_resolution=resolution
                )
                instances.append(instance)
        
        return instances


class TextContextualityAnalyzer:
    """Analyze contextuality in text using anaphoric resolution probabilities."""
    
    def __init__(self, temperature: float = 1.0):
        """Initialize analyzer.
        
        Args:
            temperature: Softmax temperature for probability extraction
        """
        self.temperature = temperature
        self._has_torch = False
        try:
            import torch
            self._has_torch = True
        except ImportError:
            pass
    
    def compute_resolution_probabilities(
        self,
        instance: AnaphoricInstance,
        llm_predict_fn=None
    ) -> Dict[str, float]:
        """Compute P(pronoun | context).
        
        Args:
            instance: AnaphoricInstance with context
            llm_predict_fn: Function that takes context and returns logit dict
            
        Returns:
            Dictionary mapping candidate to probability
        """
        if llm_predict_fn is None:
            # Dummy: uniform distribution
            return {"A": 0.5, "B": 0.5}
        
        candidates = ["he", "him", "she", "her", "they", "them"]
        logits = llm_predict_fn(instance.context, candidates)
        
        # Apply temperature scaling
        if self._has_torch:
            import torch
            logit_tensor = torch.tensor([logits.get(c, 0.0) for c in candidates])
            probs = torch.softmax(logit_tensor / self.temperature, dim=0)
            probabilities = {c: probs[i].item() for i, c in enumerate(candidates)}
        else:
            # Manual softmax
            import math
            exp_logits = [math.exp(logits.get(c, 0.0) / self.temperature) for c in candidates]
            total = sum(exp_logits)
            probabilities = {c: exp_logits[i] / total for i, c in enumerate(candidates)}
        
        # Aggregate to P(resolve to A) vs P(resolve to B)
        p_a = sum(probabilities.get(tok, 0) for tok in ["he", "him", "his"])
        p_b = sum(probabilities.get(tok, 0) for tok in ["she", "her", "hers"])
        
        # Normalize
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
        instances: List[AnaphoricInstance],
    ) -> Dict[str, List[float]]:
        """Construct CHSH measurement settings from anaphoric instances."""
        if not instances:
            return {"A": [], "A_prime": [], "B": [], "B_prime": []}
        
        if not all(hasattr(i, "probability_a") and i.probability_a is not None for i in instances):
            raise ValueError("All instances must have computed probabilities.")
        
        outcomes = {
            "A": [],
            "A_prime": [],
            "B": [],
            "B_prime": []
        }
        
        for i, instance in enumerate(instances):
            setting_idx = i % 4
            
            # Compute "spin-like" outcome: normalize probability difference to [-1, 1]
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
    
    def compute_contextuality(
        self,
        instances: List[AnaphoricInstance],
    ) -> Dict[str, Union[float, bool]]:
        """Compute CHSH S-value from anaphoric instances."""
        
        outcomes = self.construct_chsh_settings(instances)
        products = compute_chsh_products(outcomes)
        
        if not products:
            return {"error": "Could not compute CHSH products", "s_value": None}
        
        # Convert to expectation values
        expectation_values = {k: v for k, v in products.items()}
        
        # Calculate S-value
        s_value = calculate_s_value(expectation_values)
        
        # Check violations
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
        extract_from_text: Optional[str] = None,
        llm_predict_fn=None
    ) -> Dict:
        """Full analysis pipeline: generate instances, compute probabilities, calculate S-value."""
        extractor = AnaphoricSchemaExtractor()
        
        if extract_from_text:
            instances = extractor.extract_from_text(extract_from_text)
        else:
            instances = extractor.generate_instances(n_instances)
        
        # Compute probabilities for each instance
        for instance in instances:
            self.compute_resolution_probabilities(instance, llm_predict_fn)
        
        # Compute contextuality
        results = self.compute_contextuality(instances)
        results["instances"] = instances
        return results


class CorpusContextualityDataset:
    """Dataset builder for text-based contextuality analysis."""
    
    def __init__(self, min_sentence_length: int = 20, max_sentence_length: int = 200):
        self.min_length = min_sentence_length
        self.max_length = max_sentence_length
    
    def load_wikitext(self, split: str = "train") -> List[str]:
        """Load and filter WikiText sentences."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            sentences = []
            for item in dataset:
                text = item["text"].strip()
                if self.min_length <= len(text) <= self.max_length:
                    sentences.append(text)
            return sentences
        except ImportError:
            print("datasets library not installed. Install with: pip install datasets")
            return []
    
    def filter_anaphoric_sentences(self, sentences: List[str]) -> List[str]:
        """Filter sentences likely to contain anaphoric patterns."""
        patterns = [
            r"the\s+\w+\s+\w+\s+and\s+the\s+\w+\s+\w+.*?is",
            r"and\s+the\s+\w+\s+\w+.*?the\s+\w+\s+one\s+is"
        ]
        
        filtered = []
        for sent in sentences:
            for pattern in patterns:
                if re.search(pattern, sent, re.IGNORECASE):
                    filtered.append(sent)
                    break
        return filtered
