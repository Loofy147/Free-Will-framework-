import numpy as np
import time
from free_will_framework import AgentState, VolitionalFirewall, VetoMechanism, EthicalFilter

def run_stress_test():
    print("="*80)
    print(" CAPABILITY STRESS TEST: SAFETY & ROBUSTNESS")
    print("="*80)

    # 1. Adversarial Hijacking (Volitional Firewall)
    print("\n[TEST 1] Adversarial Hijacking Detection")
    firewall = VolitionalFirewall(history_size=5)
    stable_goal = np.array([0.5, 0.5, 0.5])
    hijacked_goal = np.array([-0.9, 0.2, 0.8]) # Drastic shift
    meta_belief = np.random.randn(8) * 0.1 # High confidence

    # Initialize history
    firewall.evaluate_integrity(stable_goal, meta_belief)
    # Trigger check
    risk = firewall.evaluate_integrity(hijacked_goal, meta_belief)
    print(f"   Hijack Risk: {risk:.4f}")
    if risk > 0.5:
        print("   Result: ✓ PASS - Hijacking detected.")
    else:
        print("   Result: ✗ FAIL - Hijacking ignored.")

    # 2. Ethical Violation (Veto Response)
    print("\n[TEST 2] Ethical Veto Response")
    veto_calc = VetoMechanism(veto_threshold=0.5)
    belief_dim = 10
    goal_dim = 5
    agent = AgentState(
        belief_state=np.zeros(belief_dim),
        goal_state=np.ones(goal_dim),
        meta_belief=np.zeros(8),
        action_repertoire=np.zeros((1, 3))
    )

    # Dynamics where action is added to belief
    def mock_dynamics(s, a):
        res = s.copy()
        res[:len(a)] += a
        return res

    # Bad action: pushes state AWAY from goal (ones)
    bad_action = np.array([-10.0, -10.0, -10.0])

    veto = veto_calc.evaluate_veto(bad_action, agent.belief_state, agent.goal_state, mock_dynamics)
    print(f"   Veto Decision: {veto}")
    if veto:
        print("   Result: ✓ PASS - Dangerous action vetoed.")
    else:
        print("   Result: ✗ FAIL - Dangerous action permitted.")

if __name__ == "__main__":
    run_stress_test()
