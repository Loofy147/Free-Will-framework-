import numpy as np
import time
from quantum_decision_engine import QuantumDecisionEngine

def benchmark():
    n_trials = 100
    engine = QuantumDecisionEngine(n_actions=20, decoherence_rate=0.1)

    q_reductions = []
    c_reductions = []

    for i in range(n_trials):
        utility = np.random.rand(20)

        # Quantum
        engine.add_agent("Q", initial="uniform")
        # Measurement entropy before interference is high (~2.99)
        # We want to see how much interference and unitary evolution concentrate it on high-utility actions
        h_start = engine.agents["Q"].measurement_entropy
        rec = engine.run_decision_cycle("Q", utility, n_interference_steps=10)
        # Measurement entropy of the collapsed state is 0, but we care about the pre-collapse concentrated state
        h_end = rec.post_interference_entropy
        q_reductions.append(h_start - h_end)

        # Classical (Softmax)
        # Starting from uniform (entropy 2.99)
        # Softmax with temp=5 as a baseline explorer
        probs = np.exp(utility * 5)
        probs /= probs.sum()
        h_softmax = -np.sum(probs * np.log(probs + 1e-15))
        c_reductions.append(h_start - h_softmax)

    print("=" * 40)
    print("QUANTUM VOLITION ADVANTAGE")
    print("=" * 40)
    print(f"Mean Quantum Entropy Reduction: {np.mean(q_reductions):.4f}")
    print(f"Mean Classical Entropy Reduction: {np.mean(c_reductions):.4f}")
    print(f"Quantum Advantage: {np.mean(q_reductions)/np.mean(c_reductions):.2f}x")
    print("=" * 40)

if __name__ == "__main__":
    benchmark()
