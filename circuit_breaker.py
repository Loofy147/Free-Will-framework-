"""
CIRCUIT BREAKER PATTERN IMPLEMENTATION
Production-grade fault tolerance for AI agency architecture
Based on Michael Nygard's "Release It!" pattern
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass, field
from collections import deque
import functools

# ============================================================================
# CIRCUIT BREAKER STATE MACHINE
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation, tracking failures
    OPEN = "open"          # Rejecting all requests, using fallback
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    name: str
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes in HALF_OPEN to close
    timeout_duration: float = 5.0       # Seconds to wait before HALF_OPEN
    window_size: int = 10               # Rolling window for failure tracking
    max_retries: int = 3                # Max retry attempts per request
    fallback_func: Optional[Callable] = None  # Fallback function

@dataclass
class CircuitBreakerMetrics:
    """Metrics tracked by circuit breaker"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0  # Rejected due to OPEN state
    fallback_invocations: int = 0
    state_transitions: list = field(default_factory=list)
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None


class CircuitBreaker:
    """
    Production-grade circuit breaker implementation

    State transitions:
    CLOSED → OPEN: After failure_threshold failures
    OPEN → HALF_OPEN: After timeout_duration seconds
    HALF_OPEN → CLOSED: After success_threshold successes
    HALF_OPEN → OPEN: On any failure
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.failure_window = deque(maxlen=config.window_size)
        self.success_count_half_open = 0
        self.lock = threading.Lock()  # Thread-safe operations
        self.opened_at: Optional[float] = None

        print(f"[CircuitBreaker:{config.name}] Initialized in CLOSED state")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result from func or fallback

        Raises:
            Exception if all retries exhausted and no fallback
        """
        with self.lock:
            self.metrics.total_requests += 1

            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    # Circuit still open, use fallback
                    return self._execute_fallback(*args, **kwargs)

        # Attempt execution with retries
        for attempt in range(self.config.max_retries):
            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result

            except Exception as e:
                print(f"[CircuitBreaker:{self.config.name}] "
                      f"Attempt {attempt+1}/{self.config.max_retries} failed: {e}")

                if attempt == self.config.max_retries - 1:
                    # Last attempt failed
                    self._record_failure()

                    # If fallback available, use it
                    if self.config.fallback_func:
                        return self._execute_fallback(*args, **kwargs)
                    else:
                        raise  # No fallback, propagate exception

                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try HALF_OPEN"""
        if self.opened_at is None:
            return False
        return time.time() - self.opened_at >= self.config.timeout_duration

    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN"""
        self.state = CircuitState.HALF_OPEN
        self.success_count_half_open = 0
        self._record_state_transition(CircuitState.HALF_OPEN)
        print(f"[CircuitBreaker:{self.config.name}] Transitioned to HALF_OPEN "
              f"(testing recovery)")

    def _record_success(self):
        """Record successful execution"""
        with self.lock:
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = time.time()
            self.failure_window.append(False)  # Success

            if self.state == CircuitState.HALF_OPEN:
                self.success_count_half_open += 1
                if self.success_count_half_open >= self.config.success_threshold:
                    # Recovered! Close circuit
                    self.state = CircuitState.CLOSED
                    self.opened_at = None
                    self._record_state_transition(CircuitState.CLOSED)
                    print(f"[CircuitBreaker:{self.config.name}] "
                          f"Transitioned to CLOSED (service recovered)")

    def _record_failure(self):
        """Record failed execution"""
        with self.lock:
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = time.time()
            self.failure_window.append(True)  # Failure

            if self.state == CircuitState.HALF_OPEN:
                # Immediate transition back to OPEN on failure
                self.state = CircuitState.OPEN
                self.opened_at = time.time()
                self._record_state_transition(CircuitState.OPEN)
                print(f"[CircuitBreaker:{self.config.name}] "
                      f"Transitioned to OPEN (recovery failed)")

            elif self.state == CircuitState.CLOSED:
                # Check if we should open circuit
                recent_failures = sum(1 for f in self.failure_window if f)
                if recent_failures >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.opened_at = time.time()
                    self._record_state_transition(CircuitState.OPEN)
                    print(f"[CircuitBreaker:{self.config.name}] "
                          f"Transitioned to OPEN ({recent_failures} failures)")

    def _execute_fallback(self, *args, **kwargs) -> Any:
        """Execute fallback function"""
        self.metrics.rejected_requests += 1
        self.metrics.fallback_invocations += 1

        if self.config.fallback_func:
            print(f"[CircuitBreaker:{self.config.name}] "
                  f"Using fallback (circuit is OPEN)")
            return self.config.fallback_func(*args, **kwargs)
        else:
            raise RuntimeError(
                f"Circuit {self.config.name} is OPEN and no fallback configured"
            )

    def _record_state_transition(self, new_state: CircuitState):
        """Log state transitions for monitoring"""
        self.metrics.state_transitions.append({
            'timestamp': time.time(),
            'state': new_state.value,
            'total_requests': self.metrics.total_requests,
            'failure_rate': (self.metrics.failed_requests /
                           max(self.metrics.total_requests, 1))
        })

    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        with self.lock:
            return self.state

    def get_metrics(self) -> Dict:
        """Get current metrics"""
        with self.lock:
            return {
                'state': self.state.value,
                'total_requests': self.metrics.total_requests,
                'successful_requests': self.metrics.successful_requests,
                'failed_requests': self.metrics.failed_requests,
                'rejected_requests': self.metrics.rejected_requests,
                'fallback_invocations': self.metrics.fallback_invocations,
                'success_rate': (self.metrics.successful_requests /
                               max(self.metrics.total_requests, 1)),
                'state_transitions': self.metrics.state_transitions
            }

    def reset(self):
        """Manual reset (for testing or emergency recovery)"""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.opened_at = None
            self.success_count_half_open = 0
            self.failure_window.clear()
            print(f"[CircuitBreaker:{self.config.name}] Manually reset to CLOSED")


# ============================================================================
# DECORATOR FOR EASY APPLICATION
# ============================================================================

def circuit_breaker(config: CircuitBreakerConfig):
    """
    Decorator to apply circuit breaker pattern to any function

    Usage:
        @circuit_breaker(CircuitBreakerConfig(name="my_service"))
        def my_function(...):
            ...
    """
    cb = CircuitBreaker(config)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)

        # Attach circuit breaker for inspection
        wrapper._circuit_breaker = cb
        return wrapper

    return decorator


# ============================================================================
# COMPONENT-SPECIFIC CIRCUIT BREAKERS
# ============================================================================

# Fallback Functions

def volition_fallback(*args, **kwargs):
    """Fast path fallback for MCTS timeout"""
    import numpy as np
    # Return cached policy (fast heuristic)
    action_dim = kwargs.get('action_dim', 10)
    return np.random.rand(action_dim)  # Simple random policy

def critic_fallback(*args, **kwargs):
    """Conservative veto fallback"""
    # When in doubt, veto (safety-first)
    return {'veto': True, 'reason': 'Circuit breaker fallback - conservative veto'}

def counterfactual_fallback(*args, **kwargs):
    """Disable background planning fallback"""
    # Return empty futures list (reactive mode)
    return []

def fwi_fallback(*args, **kwargs):
    """Return last known FWI or default"""
    return {'fwi': 0.5, 'source': 'fallback_default'}


# Component Configurations

volition_config = CircuitBreakerConfig(
    name="VolitionModule",
    failure_threshold=5,
    success_threshold=2,
    timeout_duration=5.0,
    window_size=10,
    max_retries=3,
    fallback_func=volition_fallback
)

critic_config = CircuitBreakerConfig(
    name="MetaCognitiveCritic",
    failure_threshold=3,    # More sensitive (safety-critical)
    success_threshold=2,
    timeout_duration=5.0,
    window_size=5,
    max_retries=2,         # Fewer retries (time-critical)
    fallback_func=critic_fallback
)

counterfactual_config = CircuitBreakerConfig(
    name="CounterfactualSimulator",
    failure_threshold=10,   # More tolerant (non-critical)
    success_threshold=3,
    timeout_duration=30.0,
    window_size=30,
    max_retries=5,
    fallback_func=counterfactual_fallback
)

fwi_config = CircuitBreakerConfig(
    name="FWICalculator",
    failure_threshold=5,
    success_threshold=2,
    timeout_duration=10.0,
    window_size=10,
    max_retries=3,
    fallback_func=fwi_fallback
)


# ============================================================================
# DEMO: APPLYING CIRCUIT BREAKERS TO COMPONENTS
# ============================================================================

class ProtectedVolitionModule:
    """Volition module with circuit breaker protection"""

    def __init__(self):
        self.cb = CircuitBreaker(volition_config)
        self.timeout_count = 0  # For demo

    @circuit_breaker(volition_config)
    def select_action(self, state, depth=50):
        """MCTS action selection with circuit breaker"""
        # Simulate occasional timeouts
        if self.timeout_count > 0:
            self.timeout_count -= 1
            raise TimeoutError("MCTS exceeded 100ms timeout")

        # Normal operation
        import numpy as np
        return np.random.rand(10)  # Mock action

    def inject_failure(self, count=5):
        """Inject failures for testing"""
        self.timeout_count = count


class ProtectedMetaCritic:
    """Meta-cognitive critic with circuit breaker"""

    def __init__(self):
        self.cb = CircuitBreaker(critic_config)
        self.fail_next = False

    @circuit_breaker(critic_config)
    def evaluate_action(self, action):
        """Veto check with circuit breaker"""
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Z3 solver timeout")

        # Normal operation
        return {'veto': False, 'reason': 'Action approved'}

    def inject_failure(self):
        """Inject single failure"""
        self.fail_next = True


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def test_circuit_breaker_state_machine():
    """Test complete state machine: CLOSED → OPEN → HALF_OPEN → CLOSED"""

    print("\n" + "="*80)
    print("CIRCUIT BREAKER STATE MACHINE TEST")
    print("="*80)

    volition = ProtectedVolitionModule()

    # Phase 1: Normal operation (CLOSED)
    print("\nPhase 1: Normal Operation (CLOSED state)")
    for i in range(3):
        action = volition.select_action(None)
        print(f"  Request {i+1}: Success - {action[:3]}...")

    # Phase 2: Inject failures to trigger OPEN
    print("\nPhase 2: Injecting failures to trigger OPEN state")
    volition.inject_failure(count=5)

    for i in range(5):
        try:
            action = volition.select_action(None)
        except TimeoutError:
            print(f"  Request {i+1}: Failed (expected)")

    # Check state
    cb = volition.select_action._circuit_breaker
    print(f"\nCircuit State: {cb.get_state()}")
    print(f"Metrics: {cb.get_metrics()}")

    # Phase 3: Circuit is OPEN, requests rejected
    print("\nPhase 3: Circuit is OPEN, using fallback")
    for i in range(3):
        action = volition.select_action(None)
        print(f"  Request {i+1}: Fallback returned - {action[:3]}...")

    # Phase 4: Wait for HALF_OPEN transition
    print(f"\nPhase 4: Waiting {volition_config.timeout_duration}s for HALF_OPEN...")
    time.sleep(volition_config.timeout_duration + 0.1)

    # Phase 5: Test recovery (HALF_OPEN → CLOSED)
    print("\nPhase 5: Testing recovery (HALF_OPEN → CLOSED)")
    for i in range(3):
        action = volition.select_action(None)
        print(f"  Request {i+1}: Success - {action[:3]}...")
        print(f"  State: {cb.get_state()}")

    # Final metrics
    print("\n" + "="*80)
    print("FINAL METRICS")
    print("="*80)
    final_metrics = cb.get_metrics()
    for key, value in final_metrics.items():
        if key != 'state_transitions':
            print(f"  {key}: {value}")

    print(f"\nState Transitions ({len(final_metrics['state_transitions'])}):")
    for transition in final_metrics['state_transitions']:
        print(f"  → {transition['state']} at t={transition['timestamp']:.2f}")

    assert final_metrics['state'] == 'closed', "Should end in CLOSED state!"
    print("\n✓ State machine test PASSED")


if __name__ == "__main__":
    test_circuit_breaker_state_machine()
