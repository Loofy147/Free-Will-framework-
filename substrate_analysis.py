import json
import numpy as np
from free_will_framework import AgentState, FreeWillIndex, BiologicalSignalSimulator

def analyze():
    with open("optimized_weights.json", "r") as f:
        weights = json.load(f)

    fwi_calc = FreeWillIndex(weights=weights)
    substrates = ['Silicon', 'Neuromorphic', 'Biotic']
    results = {}

    np.random.seed(42)
    # Generic agent
    agent = AgentState(
        belief_state=np.random.randn(10),
        goal_state=np.random.rand(5),
        meta_belief=np.random.randn(8),
        action_repertoire=np.random.randn(20, 3)
    )
    def dyn(s, a): return s * 0.9 + a.mean() * 0.1
    conn = np.eye(10)
    bounds = np.ones(3) * 2.0

    print("="*80)
    print(" CROSS-SUBSTRATE VOLITIONAL SENSITIVITY")
    print("="*80)

    for sub in substrates:
        sim = BiologicalSignalSimulator(substrate=sub)
        # 1. Compute Base FWI
        res = fwi_calc.compute(agent, dyn, conn, bounds)
        # 2. Simulate BOLD & Energy
        bold = sim.simulate_bold(res)
        energy = sim.compute_energy_cost(res)

        results[sub] = {
            'fwi': res['fwi'],
            'energy_fwi_ratio': energy['energy_fwi_ratio'],
            'bold_volition_signal': bold['global_volition_signal']
        }

        print(f"\nSubstrate: {sub}")
        print(f"   FWI:               {res['fwi']:.4f}")
        print(f"   Energy-FWI Ratio: {energy['energy_fwi_ratio']:.2e}")
        print(f"   BOLD Volition:     {bold['global_volition_signal']:.4f}")

    with open("substrate_report.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    analyze()
