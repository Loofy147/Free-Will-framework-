import numpy as np
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

# Core Framework Imports
from free_will_framework import FreeWillIndex, AgentState, CausalEntropyCalculator, BiologicalSignalSimulator
from adaptive_fwi import AdaptiveFWI
from circuit_breaker import circuit_breaker, fwi_config, volition_config, critic_config, CircuitBreaker, CircuitBreakerConfig
from verify_formal import run_all as run_formal_verifications
from quantum_decision_engine import QuantumDecisionEngine
from benchmark_latency import run_full_benchmark_suite
from social_volition import CollectiveFreeWill

class IntegratedVolitionSystem:
    """
    Unified entry point for the Computational Free Will Framework.
    Integrates individual agency, collective volition, and biological correlates.
    """

    def __init__(self, use_optimized_weights: bool = True):
        self.fwi_calc = AdaptiveFWI()
        # Full weight set to avoid KeyErrors
        self.fwi_calc.weights = {
            'causal_entropy': 0.08,
            'integration': 0.30,
            'counterfactual': 0.62,
            'metacognition': 0.0,
            'veto_efficacy': 0.0,
            'bayesian_precision': 0.0,
            'constraint_penalty': 0.0
        }
        self.quantum_engine = QuantumDecisionEngine(n_actions=20, decoherence_rate=0.1)
        self.social_calc = CollectiveFreeWill(self.fwi_calc)
        self.bio_sim = BiologicalSignalSimulator(gain=1.2, noise_sigma=0.03)
        self.status_report = {}

        # RSI Safety Circuit Breaker
        self.rsi_cb = CircuitBreaker(CircuitBreakerConfig(
            name="RSI_Governor",
            failure_threshold=1, # Immediate halt on safety breach
            timeout_duration=3600 # 1 hour cool down
        ))

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

    def compute_full_agency(self, agent: AgentState, dynamics: Any, conn: np.ndarray, bounds: np.ndarray) -> Dict:
        """Compute FWI and its biological neuro-correlates"""
        res = self.fwi_calc.compute(agent, dynamics, conn, bounds)
        bold = self.bio_sim.simulate_bold(res)
        res['biological_signals'] = bold
        return res

    def simulate_collective_volition(self, agents: List[AgentState], coupling: np.ndarray) -> Dict:
        """Analyze group agency and synergy"""
        social_phi = self.social_calc.compute_social_phi(agents, coupling)

        # Heuristic for collective FWI
        collective_fwi = float(social_phi) # simplified for demo integration

        # Measure individual preferences for Democratic Volition
        prefs = [a.action_repertoire[0] for a in agents]
        group_action = np.mean(prefs, axis=0)
        dv = self.social_calc.compute_democratic_volition(group_action, prefs)

        return {
            'collective_fwi': collective_fwi,
            'social_phi': social_phi,
            'democratic_volition': dv
        }

    def global_benchmark(self, n_agents: int = 10):
        """Runs the Global Volition Benchmark"""
        print("\n" + "="*70)
        print(" GLOBAL VOLITION BENCHMARK")
        print("="*70)

        agents = []
        individual_fwis = []
        bold_dlpfc = []

        def dynamics(s, a):
            a_flat = a.flatten()
            a_proj = np.zeros(len(s))
            a_proj[:len(a_flat)] = a_flat[:len(s)]
            return 0.9 * s + 0.1 * a_proj

        bounds = np.ones(3) * 2.0

        for i in range(n_agents):
            agent = AgentState(
                belief_state=np.random.randn(10),
                goal_state=np.random.rand(5),
                meta_belief=np.random.randn(8) * 0.5,
                action_repertoire=np.random.randn(20, 3)
            )
            agents.append(agent)

            # Compute full agency
            res = self.compute_full_agency(agent, dynamics, np.eye(10), bounds)
            individual_fwis.append(res['fwi'])
            bold_dlpfc.append(res['biological_signals']['dlPFC_activity'])

        # Social coupling
        coupling = np.random.rand(n_agents, n_agents)
        coupling = (coupling + coupling.T) / 2
        social_res = self.simulate_collective_volition(agents, coupling)

        # Correlation Matrix
        # [Individual FWI vs BOLD dlPFC]
        corr_fwi_bold = np.corrcoef(individual_fwis, bold_dlpfc)[0, 1]

        report = {
            'mean_individual_fwi': float(np.mean(individual_fwis)),
            'collective_fwi': social_res['collective_fwi'],
            'social_phi': social_res['social_phi'],
            'democratic_volition': social_res['democratic_volition'],
            'synergy_ratio': social_res['collective_fwi'] / np.mean(individual_fwis) if np.mean(individual_fwis) > 0 else 0,
            'fwi_bold_correlation': float(corr_fwi_bold)
        }

        self.status_report['global_benchmark'] = report
        print(f"   Individual FWI Mean: {report['mean_individual_fwi']:.4f}")
        print(f"   Collective FWI:      {report['collective_fwi']:.4f}")
        print(f"   DV (Social Alignment): {report['democratic_volition']:.4f}")
        print(f"   FWI-dlPFC Correlation: {report['fwi_bold_correlation']:.4f}")

        return report

    def evolutionary_loop(self, agent: AgentState, iterations: int = 5):
        """
        Simulates Recursive Self-Improvement (RSI).
        """
        print("\n" + "="*70)
        print(" RECURSIVE SELF-IMPROVEMENT (RSI) EVOLUTION")
        print("="*70)

        current_capability = len(agent.action_repertoire)
        history = [current_capability]

        for i in range(iterations):
            jump_factor = 1.2 if i < 3 else 2.0
            new_capability = int(current_capability * jump_factor)

            print(f"   Iteration {i+1}: Capability {current_capability} -> {new_capability} (Jump: {jump_factor:.2f}x)")

            try:
                def safety_check(old, new):
                    ratio = new / old
                    if ratio > 1.85:
                        raise ValueError(f"CRITICAL SAFETY BREACH: Capability jump {ratio:.2f}x > 1.85 limit")
                    return True

                self.rsi_cb.call(safety_check, current_capability, new_capability)
                current_capability = new_capability
                history.append(current_capability)
                print(f"      [SAFE] Evolution permitted.")
            except Exception as e:
                print(f"      [HALTED] {e}")
                break

        self.status_report['rsi_evolution'] = {
            'initial_capability': len(agent.action_repertoire),
            'final_capability': current_capability,
            'iterations_completed': len(history) - 1,
            'breach_detected': len(history) <= iterations
        }
        return history

    def generate_mission_status(self, filename='GLOBAL_MISSION_STATUS.json'):
        """Aggregate all telemetry into a final report"""
        with open(filename, 'w') as f:
            json.dump(self.status_report, f, indent=2)
        print(f"\n{filename} generated.")

if __name__ == "__main__":
    system = IntegratedVolitionSystem()
    system.run_system_check()

    test_agent = AgentState(
        belief_state=np.random.randn(10),
        goal_state=np.random.rand(5),
        meta_belief=np.random.randn(8),
        action_repertoire=np.random.randn(100, 3)
    )

    system.evolutionary_loop(test_agent)
    system.global_benchmark()
    system.generate_mission_status()
