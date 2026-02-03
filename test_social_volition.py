"""
UNIT TESTS FOR SOCIAL VOLITION (P6)
"""

import numpy as np
import time
from social_volition import SwarmSimulator, CollectiveFreeWill
from free_will_framework import FreeWillIndex

def test_social_scalability():
    """Verify that social FWI computation scales reasonably with M agents"""
    fwi_calc = FreeWillIndex()
    social_calc = CollectiveFreeWill(fwi_calc)

    sizes = [10, 50, 100]
    times = []

    for m in sizes:
        simulator = SwarmSimulator(n_agents=m)
        start = time.time()
        # Run one step
        simulator.run_step(coupling_strength=0.5)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  M={m:3d} agents: {elapsed:.4f}s")

    # Check that it's not exponential
    assert times[2] < times[1] * 5, "Scalability looks poor!"
    print("  Social scalability verified ✓")

def test_robustness_to_noise():
    """Collective FWI should maintain some integrity despite noisy agents"""
    simulator = SwarmSimulator(n_agents=50)

    # Baseline
    res_clean = simulator.run_step(coupling_strength=0.8, noise_level=0.0)

    # Higher noise should decrease FWI but not crash it
    res_noisy = simulator.run_step(coupling_strength=0.8, noise_level=0.5)

    print(f"  Clean FWI: {res_clean['collective_fwi']:.4f}")
    print(f"  Noisy FWI: {res_noisy['collective_fwi']:.4f}")

    assert res_noisy['collective_fwi'] > 0.1, "Group agency completely collapsed under noise!"
    print("  Robustness verified ✓")

def test_democratic_volition():
    """Verify that DV correctly measures alignment"""
    fwi_calc = FreeWillIndex()
    social_calc = CollectiveFreeWill(fwi_calc)

    # Case 1: Perfect alignment
    group_action = np.array([1.0, 0.0, 0.0])
    prefs = [np.array([1.0, 0.0, 0.0])] * 10
    dv_perfect = social_calc.compute_democratic_volition(group_action, prefs)

    # Case 2: Random alignment
    prefs_random = [np.random.randn(3) for _ in range(10)]
    dv_random = social_calc.compute_democratic_volition(group_action, prefs_random)

    print(f"  Perfect DV: {dv_perfect:.4f}")
    print(f"  Random DV:  {dv_random:.4f}")

    assert dv_perfect > 0.99
    assert dv_perfect > dv_random
    print("  Democratic volition verified ✓")

if __name__ == "__main__":
    print("="*80)
    print("SOCIAL VOLITION UNIT TESTS")
    print("="*80)

    try:
        test_social_scalability()
        test_robustness_to_noise()
        test_democratic_volition()
        print("\nALL SOCIAL VOLITION TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
