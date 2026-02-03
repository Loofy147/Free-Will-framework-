import numpy as np
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Any

# Core Framework Imports
from free_will_framework import FreeWillIndex, AgentState, CausalEntropyCalculator
from adaptive_fwi import AdaptiveFWI
from circuit_breaker import circuit_breaker, fwi_config, volition_config, critic_config
from verify_formal import run_all as run_formal_verifications
from quantum_decision_engine import QuantumDecisionEngine
from benchmark_latency import run_full_benchmark_suite

class IntegratedVolitionSystem:
    """
    Unified entry point for the Computational Free Will Framework.
    Integrates optimization, formal verification, safety monitoring, and quantum extensions.
    """

    def __init__(self, use_optimized_weights: bool = True):
        self.fwi_calc = AdaptiveFWI()
        if use_optimized_weights:
            # Load optimized weights found in step 2
            self.fwi_calc.weights = {
                'causal_entropy': 0.0800,
                'integration': 0.3000,
                'counterfactual': 0.6200,
                'metacognition': 0.0,
                'constraint_penalty': 0.0
            }
        self.quantum_engine = QuantumDecisionEngine(n_actions=20, decoherence_rate=0.1)
        self.status_report = {}

    def run_system_check(self) -> bool:
        """Run formal verification and latency benchmarks"""
        print("\n[SYSTEM CHECK] Running Formal Verifications...")
        formal_results = run_formal_verifications()
        all_formal_passed = all(r.status == "VERIFIED" for r in formal_results)

        print(f"\n[SYSTEM CHECK] Running Latency Benchmarks...")
        latency_results = run_full_benchmark_suite()
        all_latency_passed = all(r.passed for r in latency_results.values())

        self.status_report['formal_verification'] = [r.to_dict() for r in formal_results]
        self.status_report['latency_benchmarks'] = {k: v.to_dict() for k, v in latency_results.items()}
        self.status_report['healthy'] = all_formal_passed and all_latency_passed

        return self.status_report['healthy']

    @circuit_breaker(fwi_config)
    def compute_agency_score(self, agent: AgentState, dynamics: Any) -> Dict:
        """Compute FWI with safety circuit breaker protection"""
        # Internal call to the framework
        # Mocking the actual call for integration demonstration
        fwi_val = self.fwi_calc.compute_fwi(agent, dynamics)
        return fwi_val

    def execute_quantum_decision(self, agent_id: str, utilities: np.ndarray) -> Dict:
        """Execute a quantum-accelerated decision cycle"""
        if agent_id not in self.quantum_engine.agents:
            self.quantum_engine.add_agent(agent_id)

        record = self.quantum_engine.run_decision_cycle(agent_id, utilities)
        return {
            'action': record.final_action,
            'entropy_reduction': record.initial_entropy - record.post_decoherence_entropy,
            'duration_ms': record.duration_ms
        }

    def generate_mission_status(self):
        """Aggregate all telemetry into a final report"""
        with open('MISSION_STATUS.json', 'w') as f:
            json.dump(self.status_report, f, indent=2)
        print("\nMISSION_STATUS.json generated.")

if __name__ == "__main__":
    system = IntegratedVolitionSystem()
    print("=" * 70)
    print(" INTEGRATED VOLITION SYSTEM INITIALIZED")
    print("=" * 70)

    # 1. Verify system integrity
    healthy = system.run_system_check()
    print(f"\nSystem Healthy: {'YES' if healthy else 'NO (Verification/Latency Failure)'}")

    # 2. Demo Quantum Decision
    print("\n[DEMO] Executing Quantum Decision...")
    u = np.random.rand(20)
    res = system.execute_quantum_decision("Agent_Alpha", u)
    print(f"       Action: {res['action']} | Entropy Red: {res['entropy_reduction']:.4f} | Latency: {res['duration_ms']:.2f}ms")

    # 3. Final Report
    system.generate_mission_status()
