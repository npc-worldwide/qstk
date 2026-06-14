#!/usr/bin/env python3
"""NumPy-only demonstration of CHSH Bell test violation.

This script demonstrates the CHSH inequality violation without requiring
PyTorch or any quantum simulation libraries. It shows:

1. The classical bound: |S| ≤ 2
2. The quantum maximum (Tsirelson bound): |S| = 2√2 ≈ 2.828
3. How to simulate Bell test measurements

Example usage:
    python chsh_numpy_demo.py

Reference:
    Lo, Sadrzadeh, Mansfield (2024). "Quantum-Like Contextuality in 
    Large Language Models." arXiv:2412.16806
"""

import numpy as np
from typing import Dict, Tuple


class CHSHSimulator:
    """
    Simulate CHSH Bell test with Bell state |Φ+⟩.
    
    The CHSH inequality is:
        S = E(a,b) + E(a,b') + E(a',b) - E(a',b')
    
    Classical bound: |S| ≤ 2
    Quantum maximum: |S| = 2√2 ≈ 2.828 (Tsirelson bound)
    
    For Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2, the correlation is:
        E(a,b) = cos(a - b)
    
    Optimal angles:
        Alice: a = 0, a' = π/2
        Bob:   b = π/4, b' = 3π/4 (or -π/4)
    """
    
    def __init__(self):
        # Optimal CHSH angles
        self.a = 0              # Alice setting A
        self.ap = np.pi/2       # Alice setting A'  
        self.b = np.pi/4        # Bob setting B
        self.bp = 3*np.pi/4     # Bob setting B' (or -π/4)
        
        # Tsirelson bound
        self.tsirelson = 2 * np.sqrt(2)
    
    def correlation(self, angle_a: float, angle_b: float) -> float:
        """
        Quantum correlation for Bell state |Φ+⟩.
        
        E(a,b) = cos(a - b) for |Φ+⟩
        """
        return np.cos(angle_a - angle_b)
    
    def theoretical_correlations(self) -> Dict[str, float]:
        """Return theoretical optimal correlations."""
        return {
            "E(a,b)":   self.correlation(self.a, self.b),
            "E(a,b')":  self.correlation(self.a, self.bp),
            "E(a',b)":  self.correlation(self.ap, self.b),
            "E(a',b')": self.correlation(self.ap, self.bp),
        }
    
    def theoretical_s_value(self) -> float:
        """Calculate theoretical S-value."""
        E = self.theoretical_correlations()
        return (E["E(a,b)"] + E["E(a,b')"] + E["E(a',b)"] - E["E(a',b')"])
    
    def simulate_measurement(self, angle_a: float, angle_b: float, 
                           n_trials: int = 1000) -> float:
        """
        Simulate Bell test measurement at given angles.
        
        Args:
            angle_a: Alice's measurement angle
            angle_b: Bob's measurement angle
            n_trials: Number of measurement trials
            
        Returns:
            Estimated correlation E(a,b)
        """
        # Expected correlation
        expected_corr = self.correlation(angle_a, angle_b)
        
        # Probability of matching outcomes
        p_match = (1 + expected_corr) / 2
        
        products = []
        for _ in range(n_trials):
            # Alice's outcome (+1 or -1)
            outcome_a = 1 if np.random.random() < 0.5 else -1
            
            # Bob's outcome (correlated with Alice's)
            if np.random.random() < p_match:
                outcome_b = outcome_a  # Match
            else:
                outcome_b = -outcome_a  # Anti-match
            
            products.append(outcome_a * outcome_b)
        
        return np.mean(products)
    
    def run_experiment(self, n_trials: int = 10000) -> Dict:
        """
        Run full CHSH Bell test experiment.
        
        Args:
            n_trials: Number of trials per setting
            
        Returns:
            Dictionary with results
        """
        print("=" * 70)
        print("CHSH Bell Test Simulation (NumPy Only)")
        print("=" * 70)
        print()
        print("Bell state: |Φ+⟩ = (|00⟩ + |11⟩)/√2")
        print()
        
        # Theoretical values
        print("Theoretical correlations:")
        theory = self.theoretical_correlations()
        for name, val in theory.items():
            print(f"  {name} = {val:.4f}")
        print()
        
        s_theory = self.theoretical_s_value()
        print(f"Theoretical S-value: {s_theory:.4f}")
        print(f"                   |S| = {abs(s_theory):.4f}")
        print()
        
        # Simulate measurements
        print(f"Simulating {n_trials} trials per setting...")
        print()
        
        results = {}
        settings = [
            ("E(a,b)",   self.a,   self.b),
            ("E(a,b')",  self.a,   self.bp),
            ("E(a',b)",  self.ap,  self.b),
            ("E(a',b')", self.ap,  self.bp),
        ]
        
        for name, angle_a, angle_b in settings:
            results[name] = self.simulate_measurement(angle_a, angle_b, n_trials)
        
        print("Simulated correlations:")
        for name, val in results.items():
            print(f"  {name} = {val:.4f}")
        print()
        
        # Calculate S-value from simulation
        s_sim = (results["E(a,b)"] + results["E(a,b')"] + 
                results["E(a',b)"] - results["E(a',b')"])
        
        print(f"Simulated S-value: {s_sim:.4f}")
        print(f"                 |S| = {abs(s_sim):.4f}")
        print()
        
        # Check bounds
        print("Bounds:")
        print(f"  Classical:     |S| ≤ 2.0")
        print(f"  Tsirelson:     |S| ≤ 2√2 ≈ {self.tsirelson:.4f}")
        print()
        
        # Determine violation
        classical_violation = abs(s_sim) > 2.0
        tsirelson_violation = abs(s_sim) > self.tsirelson + 0.01
        
        if tsirelson_violation:
            print("⚠ Warning: S exceeds Tsirelson bound (numerical error)")
        elif classical_violation:
            print(f"✓✓✓ QUANTUM VIOLATION!")
            print(f"    |S| = {abs(s_sim):.4f} > 2.0 (classical bound)")
            print(f"    This demonstrates quantum contextuality!")
            print()
            print(f"    Violation magnitude: {abs(s_sim) - 2.0:.4f}")
            print(f"    Fraction of quantum max: {abs(s_sim)/self.tsirelson:.1%}")
        else:
            print(f"✗ No violation: |S| = {abs(s_sim):.4f} ≤ 2.0")
            print(f"  (Try running with more trials: --trials 50000)")
        
        return {
            "theoretical_s": s_theory,
            "simulated_s": s_sim,
            "theoretical_correlations": theory,
            "simulated_correlations": results,
            "classical_violation": classical_violation,
            "tsirelson_violation": tsirelson_violation,
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CHSH Bell Test Simulation (NumPy Only)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10000,
        help="Number of measurement trials per setting (default: 10000)"
    )
    
    args = parser.parse_args()
    
    # Run simulation
    simulator = CHSHSimulator()
    results = simulator.run_experiment(n_trials=args.trials)
    
    print()
    print("=" * 70)
    print("Interpretation")
    print("=" * 70)
    print()
    print("The CHSH inequality |S| ≤ 2 is a fundamental limit on")
    print("correlations for any classical system satisfying local")
    print("realism (no faster-than-light communication, predetermined")
    print("outcomes independent of measurement choice).")
    print()
    print("Quantum mechanics violates this bound because entangled")
    print("particles can exhibit correlations that exceed classical")
    print("limits. The maximum quantum violation is |S| = 2√2,")
    print("known as the Tsirelson bound.")
    print()
    print("Lo et al. (2024) found BERT achieves S ≈ 2.2, which")
    print(f"exceeds the classical bound and reaches {2.2/(2*np.sqrt(2)):.1%}")
    print("of the quantum maximum - evidence of quantum-like")
    print("contextuality in language models!")
    print()


if __name__ == "__main__":
    main()
