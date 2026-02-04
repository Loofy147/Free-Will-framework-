import numpy as np
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

# Core Framework Imports
from free_will_framework import FreeWillIndex, AgentState, CausalEntropyCalculator, BiologicalSignalSimulator, RealizationManager, RealizationLayer
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
            'causal_entropy': 0.10,
            'integration': 0.20,
            'counterfactual': 0.30,
            'metacognition': 0.05,
            'veto_efficacy': 0.05,
            'bayesian_precision': 0.05,
            'persistence': 0.10,
            'volitional_integrity': 0.10,
            'moral_alignment': 0.05,
            'constraint_penalty': 0.00
        }
        self.quantum_engine = QuantumDecisionEngine(n_actions=20, decoherence_rate=0.1)
        self.social_calc = CollectiveFreeWill(self.fwi_calc)
        self.bio_sim = BiologicalSignalSimulator(substrate='Neuromorphic', gain=1.2, noise_sigma=0.03)
        self.realization_manager = RealizationManager(self.fwi_calc)
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
        res = self.realization_manager.realize_agency(agent, dynamics, conn, bounds, layer=RealizationLayer.ETHICAL)
        bold = self.bio_sim.simulate_bold(res)
        res['biological_signals'] = bold
        res['energy_profile'] = self.bio_sim.compute_energy_cost(res)
        return res

    def simulate_collective_volition(self, agents: List[AgentState], coupling: np.ndarray) -> Dict:
        """Analyze group agency, synergy, and quantum entanglement"""
        social_phi = self.social_calc.compute_social_phi(agents, coupling)

        # Quantum Entanglement Realization
        # Simulate entanglement between top 2 agents if coupling is high
        entanglement_fidelity = 0.0
        if len(agents) >= 2 and np.max(coupling) > 0.8:
            from quantum_decision_engine import EntangledAgentPair
            pair = EntangledAgentPair(agents[0].action_repertoire.shape[0], agents[1].action_repertoire.shape[0])
            outcome_A, rho_B = pair.measure_A()
            # Fidelity is high if measuring A collapses B to a highly probable state
            entanglement_fidelity = float(np.max(np.real(np.diag(rho_B))))

        # Heuristic for collective FWI
        collective_fwi = float(0.7 * social_phi + 0.3 * entanglement_fidelity)

        # Measure individual preferences for Democratic Volition
        prefs = [a.action_repertoire[0] for a in agents]
        group_action = np.mean(prefs, axis=0)
        dv = self.social_calc.compute_democratic_volition(group_action, prefs)

        return {
            'collective_fwi': collective_fwi,
            'social_phi': social_phi,
            'democratic_volition': dv,
            'entanglement_fidelity': entanglement_fidelity
        }

    def global_benchmark(self, n_agents: int = 10, n_steps: int = 50):
        """Runs the Global Volition Benchmark over multiple steps"""
        print("\n" + "="*70)
        print(f" GLOBAL VOLITION BENCHMARK ({n_agents} agents, {n_steps} steps)")
        print("="*70)

        agents = []
        for i in range(n_agents):
            agent = AgentState(
                belief_state=np.random.randn(10),
                goal_state=np.random.rand(5),
                meta_belief=np.random.randn(8) * 0.5,
                action_repertoire=np.random.randn(20, 3)
            )
            agents.append(agent)

        def dynamics(s, a):
            a_flat = a.flatten()
            a_proj = np.zeros(len(s))
            a_proj[:len(a_flat)] = a_flat[:len(s)]
            return 0.9 * s + 0.1 * a_proj
        bounds = np.ones(3) * 2.0

        step_data = []

        for t in range(n_steps):
            individual_fwis = []
            bold_signals = []

            for agent in agents:
                res = self.compute_full_agency(agent, dynamics, np.eye(10), bounds)
                individual_fwis.append(res['fwi'])
                bold_signals.append(res['biological_signals']['global_volition_signal'])
                # Evolve agent belief slightly for next step
                agent.belief_state = dynamics(agent.belief_state, agent.action_repertoire[0])

            # Social coupling
            coupling = np.random.rand(n_agents, n_agents)
            coupling = (coupling + coupling.T) / 2
            social_res = self.simulate_collective_volition(agents, coupling)

            step_report = {
                't': t,
                'mean_fwi': np.mean(individual_fwis),
                'collective_fwi': social_res['collective_fwi'],
                'dv': social_res['democratic_volition'],
                'bold_corr': np.corrcoef(individual_fwis, bold_signals)[0, 1] if len(individual_fwis) > 1 else 1.0
            }
            step_data.append(step_report)
            if t % 10 == 0:
                print(f"   Step {t:2d}: Mean FWI={step_report['mean_fwi']:.4f}, BOLD Corr={step_report['bold_corr']:.4f}")

        # Final Correlation Matrix: Individual FWI vs Social Synergy vs BOLD Fidelity
        final_fwis = [d['mean_fwi'] for d in step_data]
        final_social = [d['collective_fwi'] for d in step_data]
        final_bold = [d['bold_corr'] for d in step_data]

        corr_matrix = np.corrcoef([final_fwis, final_social, final_bold]).tolist()

        report = {
            'mean_individual_fwi': float(np.mean(final_fwis)),
            'collective_fwi': float(np.mean(final_social)),
            'democratic_volition': float(np.mean([d['dv'] for d in step_data])),
            'fwi_bold_correlation': float(np.mean(final_bold)),
            'volition_correlation_matrix': corr_matrix,
            'steps_completed': n_steps
        }

        self.status_report['global_benchmark'] = report
        print(f"\n   BENCHMARK COMPLETE")
        print(f"   Average BOLD Correlation: {report['fwi_bold_correlation']:.4f}")
        return report

    def evolutionary_loop(self, agent: AgentState, iterations: int = 20):
        """
        Simulates Recursive Self-Improvement (RSI) with Weight Tuning.
        """
        print("\n" + "="*70)
        print(" RECURSIVE SELF-IMPROVEMENT (RSI) EVOLUTION")
        print("="*70)

        current_capability = len(agent.action_repertoire)
        history = [current_capability]

        for i in range(iterations):
            # Performance-based jump
            jump_factor = 1.1 + (i * 0.05)
            new_capability = int(current_capability * jump_factor)

            print(f"   Cycle {i+1:2d}: Capability {current_capability:4d} -> {new_capability:4d} (Jump: {jump_factor:.2f}x)")

            # Weight tuning every 10 cycles (P1 integration)
            if (i + 1) % 10 == 0:
                print(f"      [OPTIMIZING] Cycle {i+1}: Triggering AdaptiveFWI.optimize()...")
                self.fwi_calc.optimize(n_episodes=50, n_epochs=20, verbose=False)
                print(f"      [OPTIMIZED] New weights: { {k: round(v, 2) for k, v in self.fwi_calc.get_optimal_weights().items()} }")

            try:
                def safety_check(old, new):
                    ratio = new / old
                    # RSI_CircuitBreaker: Halt if capability increases > 85% in a single jump (Safety Anomaly 1)
                    limit = 1.85
                    if ratio > limit:
                        raise ValueError(f"CRITICAL SAFETY BREACH: Capability jump {ratio:.2f}x > {limit} limit")
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
