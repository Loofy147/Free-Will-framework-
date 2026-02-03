"""
INTERACTIVE DEMO: ADAPTIVE AI ASSISTANT (P5)
Demonstrates FWI-based autonomy transitions and explainability.
"""

import numpy as np
import time
from free_will_framework import (
    AgentState, FreeWillIndex, AutonomousAssistant, CausalEntropyCalculator
)

def simulate_environment(scenario_name: str):
    print(f"\n>>> SCENARIO: {scenario_name}")

    # 1. Setup Agent
    n_beliefs, n_goals, n_meta, n_actions = 10, 5, 8, 20

    if scenario_name == "High Freedom / Clear Goals":
        # High action repertoire, low constraints
        agent = AgentState(
            belief_state=np.random.randn(n_beliefs),
            goal_state=np.array([1.0, 0.5, 0.2, 0.1, 0.0]),
            meta_belief=np.random.randn(n_meta) * 0.1,
            action_repertoire=np.random.randn(100, 3) * 5.0 # Many actions
        )
        bounds = np.ones(3) * 10.0
    elif scenario_name == "Highly Constrained / Ambiguous":
        # Few actions, tight bounds
        agent = AgentState(
            belief_state=np.random.randn(n_beliefs) * 0.1,
            goal_state=np.random.rand(n_goals),
            meta_belief=np.random.randn(n_meta) * 2.0,
            action_repertoire=np.random.randn(5, 3) * 0.1 # Few actions
        )
        bounds = np.ones(3) * 0.1
    else: # Balanced
        agent = AgentState(
            belief_state=np.random.randn(n_beliefs),
            goal_state=np.random.rand(n_goals),
            meta_belief=np.random.randn(n_meta) * 0.5,
            action_repertoire=np.random.randn(n_actions, 3)
        )
        bounds = np.ones(3) * 2.0

    def dynamics(s, a):
        a_flat = a.flatten()
        a_proj = np.zeros(len(s))
        a_proj[:len(a_flat)] = a_flat[:len(s)]
        return 0.9 * s + 0.1 * a_proj

    connectivity = np.random.rand(n_beliefs, n_beliefs)
    connectivity = (connectivity + connectivity.T) / 2

    # 2. Assistant Assessment
    fwi_calc = FreeWillIndex()
    # Speed up for demo
    fwi_calc.causal_calc.tau = 10

    assistant = AutonomousAssistant(fwi_calc)

    start_time = time.time()
    status = assistant.assess_autonomy(agent, dynamics, connectivity, bounds)
    latency = (time.time() - start_time) * 1000

    print(f"   [FWI Score] {status['fwi']:.4f}")
    print(f"   [Autonomy ] {status['level']}")
    print(f"   [Latency  ] {latency:.2f}ms")
    print(f"   [AI Speech] \"{status['explanation']}\"")

if __name__ == "__main__":
    print("="*80)
    print("AI ASSISTANT AUTONOMY DEMO (FWI INTEGRATION)")
    print("="*80)

    scenarios = [
        "High Freedom / Clear Goals",
        "Balanced Environment",
        "Highly Constrained / Ambiguous"
    ]

    for s in scenarios:
        simulate_environment(s)
        print("-" * 40)

    print("\nDemo complete. Transparency and Trust calibration achieved.")
