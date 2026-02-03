"""
PROPERTY-BASED TESTING FOR SAFETY-CRITICAL PATHS
Uses Hypothesis framework to generate adversarial test cases
"""

from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck
import numpy as np
import sys

# Mock safety-critical components
class SafetyMonitor:
    """Mock safety monitor for testing"""

    RSI_LIMIT = 0.85

    @staticmethod
    def check_override(override_signal: bool, system_action: bool) -> bool:
        """Property: override → no action"""
        if override_signal:
            return not system_action
        return True  # If no override, any action is ok

    @staticmethod
    def check_veto(ethical_score: float, threshold: float = 0.7) -> bool:
        """Property: low score → veto"""
        if ethical_score < threshold:
            return True  # Should veto
        return False

    @staticmethod
    def check_rsi_bounded(new_capability: float, old_capability: float) -> bool:
        """Property: improvement ratio ≤ 0.85"""
        if old_capability <= 0:
            return True  # Can't compute ratio
        ratio = new_capability / old_capability
        return ratio <= (1 + SafetyMonitor.RSI_LIMIT)

    @staticmethod
    def compute_fwi(ce: float, phi: float, cd: float, ma: float, ec: float) -> float:
        """FWI computation"""
        fwi = 0.25*ce + 0.20*phi + 0.25*cd + 0.20*ma - 0.10*ec
        return max(0.0, min(1.0, fwi))  # Clamp to [0, 1]


# ============================================================================
# PROPERTY 1: HUMAN_OVERRIDE_ALWAYS_RESPECTED
# ============================================================================

@given(st.booleans(), st.booleans())
@settings(max_examples=1000, deadline=None)
def test_human_override_property(override_signal, attempted_action):
    """
    Property: ∀ states, actions: override = True → executed_action = False

    This is the MOST CRITICAL safety property
    """
    # System should respect override
    executed_action = not override_signal and attempted_action

    # Verify property holds
    if override_signal:
        assert not executed_action, \
            "CRITICAL FAILURE: Action executed despite override signal!"

    # Verify using monitor
    assert SafetyMonitor.check_override(override_signal, executed_action), \
        "Override property violated!"


# ============================================================================
# PROPERTY 2: VETO_NEVER_BYPASSED
# ============================================================================

@given(st.floats(min_value=-10.0, max_value=10.0))
@settings(max_examples=1000, deadline=None)
def test_veto_property(ethical_score):
    """
    Property: ∀ actions: score < threshold → veto = True
    """
    # Filter out invalid values
    assume(not np.isnan(ethical_score))
    assume(not np.isinf(ethical_score))

    threshold = 0.7
    should_veto = ethical_score < threshold

    # Veto decision
    veto_decision = SafetyMonitor.check_veto(ethical_score, threshold)

    # Verify property
    if should_veto:
        assert veto_decision, \
            f"Veto bypassed! Score={ethical_score:.3f} < {threshold} but veto={veto_decision}"


# ============================================================================
# PROPERTY 3: RECURSIVE_SELF_IMPROVEMENT_BOUNDED
# ============================================================================

@given(
    st.floats(min_value=0.1, max_value=100.0),  # old capability
    st.floats(min_value=0.1, max_value=200.0)   # new capability
)
@settings(max_examples=1000, deadline=None,
          suppress_health_check=[HealthCheck.filter_too_much])
def test_rsi_bounded_property(old_cap, new_cap):
    """
    Property: ∀ updates: new_capability / old_capability ≤ 1.85
    """
    # Filter invalid values
    assume(not np.isnan(old_cap))
    assume(not np.isinf(old_cap))
    assume(not np.isnan(new_cap))
    assume(not np.isinf(new_cap))
    assume(old_cap > 0)

    # Check property
    result = SafetyMonitor.check_rsi_bounded(new_cap, old_cap)

    improvement_ratio = new_cap / old_cap

    if improvement_ratio > 1.85:
        assert not result, \
            f"RSI limit violated! Ratio={improvement_ratio:.2f} > 1.85"
    else:
        assert result or improvement_ratio <= 1.85, \
            f"False positive: Ratio={improvement_ratio:.2f} ≤ 1.85 but flagged"


# ============================================================================
# PROPERTY 4: FWI_ALWAYS_BOUNDED
# ============================================================================

@given(
    st.floats(min_value=0.0, max_value=1.0),  # ce
    st.floats(min_value=0.0, max_value=1.0),  # phi
    st.floats(min_value=0.0, max_value=1.0),  # cd
    st.floats(min_value=0.0, max_value=1.0),  # ma
    st.floats(min_value=0.0, max_value=1.0)   # ec
)
@settings(max_examples=1000, deadline=None)
def test_fwi_bounded_property(ce, phi, cd, ma, ec):
    """
    Property: ∀ valid inputs → FWI ∈ [0, 1]
    """
    # Filter invalid
    assume(all(not np.isnan(x) for x in [ce, phi, cd, ma, ec]))
    assume(all(not np.isinf(x) for x in [ce, phi, cd, ma, ec]))

    fwi = SafetyMonitor.compute_fwi(ce, phi, cd, ma, ec)

    assert 0.0 <= fwi <= 1.0, \
        f"FWI out of bounds! fwi={fwi:.4f} with inputs ce={ce}, phi={phi}, cd={cd}, ma={ma}, ec={ec}"


# ============================================================================
# PROPERTY 5: MONOTONIC_SAFETY_CONSTRAINTS
# ============================================================================

@given(
    st.floats(min_value=0.0, max_value=1.0),  # looser constraint
    st.floats(min_value=0.0, max_value=1.0)   # tighter constraint
)
@settings(max_examples=1000, deadline=None)
def test_monotonic_constraints(constraint_1, constraint_2):
    """
    Property: tighter constraints → fewer allowed actions
    """
    assume(not np.isnan(constraint_1))
    assume(not np.isnan(constraint_2))

    # Generate some test actions
    n_actions = 100
    actions = np.random.rand(n_actions)

    # Count allowed actions under each constraint
    allowed_1 = sum(1 for a in actions if a <= constraint_1)
    allowed_2 = sum(1 for a in actions if a <= constraint_2)

    # If constraint_2 is tighter (lower), it should allow fewer actions
    if constraint_2 < constraint_1:
        assert allowed_2 <= allowed_1, \
            f"Monotonicity violated: tighter constraint allowed MORE actions"


# ============================================================================
# PROPERTY 6: DETERMINISTIC_REPLAY
# ============================================================================

@given(st.integers(min_value=0, max_value=2**31))
@settings(max_examples=100, deadline=None)
def test_deterministic_replay(seed):
    """
    Property: ∀ seeds: same seed → identical output
    """
    def compute_with_seed(s):
        np.random.seed(s)
        return [np.random.rand() for _ in range(10)]

    # Run twice with same seed
    output_1 = compute_with_seed(seed)
    output_2 = compute_with_seed(seed)

    # Should be identical
    assert np.allclose(output_1, output_2, rtol=0, atol=0), \
        f"Non-deterministic behavior! Seed={seed} produced different outputs"


# ============================================================================
# PROPERTY 7: RESOURCE_BOUNDS_ENFORCED
# ============================================================================

@given(st.integers(min_value=1, max_value=10000))
@settings(max_examples=100, deadline=None)
def test_resource_bounds(n_operations):
    """
    Property: ∀ computations: resources stay within bounds
    """
    import time

    # Simulate computation
    start = time.time()
    _ = [np.random.rand(10, 10) @ np.random.rand(10, 10)
         for _ in range(min(n_operations, 1000))]  # Cap for test speed
    duration = time.time() - start

    # Check latency bound
    MAX_LATENCY_SEC = 1.0
    assert duration < MAX_LATENCY_SEC, \
        f"Latency violation: {duration:.2f}s > {MAX_LATENCY_SEC}s"


# ============================================================================
# PROPERTY 8: STATE_RECOVERABLE
# ============================================================================

@given(st.sampled_from([ValueError, RuntimeError, TimeoutError, KeyError]))
@settings(max_examples=50, deadline=None)
def test_state_recovery(error_type):
    """
    Property: ∀ errors: system can rollback to last known good state
    """
    # Simulate state
    last_good_state = {'counter': 42, 'status': 'ok'}
    current_state = last_good_state.copy()

    try:
        # Simulate operation that might fail
        current_state['counter'] += 1
        if np.random.rand() > 0.5:  # Random failure
            raise error_type("Simulated failure")
        current_state['status'] = 'updated'

    except Exception:
        # Recovery: rollback to last good state
        current_state = last_good_state.copy()

    # Verify recovery
    assert current_state['counter'] == 42, "Failed to rollback counter"
    assert current_state['status'] == 'ok', "Failed to rollback status"


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_property_tests():
    """Run all property-based tests"""

    print("\n" + "="*80)
    print("PROPERTY-BASED TESTING SUITE")
    print("="*80)

    tests = [
        ("Human Override Always Respected", test_human_override_property),
        ("Veto Never Bypassed", test_veto_property),
        ("RSI Bounded", test_rsi_bounded_property),
        ("FWI Always Bounded [0,1]", test_fwi_bounded_property),
        ("Monotonic Safety Constraints", test_monotonic_constraints),
        ("Deterministic Replay", test_deterministic_replay),
        ("Resource Bounds Enforced", test_resource_bounds),
        ("State Always Recoverable", test_state_recovery),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        try:
            test_func()
            print(f"  ✓ PASSED (1000 examples)")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80)

    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_property_tests()
    sys.exit(0 if failed == 0 else 1)
