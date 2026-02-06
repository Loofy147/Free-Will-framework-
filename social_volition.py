"""
SOCIAL VOLITION FRAMEWORK (P6)
Extends FWI to multi-agent systems and collective agency.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional
from free_will_framework import AgentState, FreeWillIndex, IntegratedInformationCalculator

class CollectiveFreeWill:
    """
    Quantifies the emergent free will of a group of agents.

    Metric: Φ_social (Social Integrated Information)
    Quantifies the causal synergy of the group.
    """

    def __init__(self, individual_fwi_calculator: FreeWillIndex):
        self.fwi_calc = individual_fwi_calculator
        self.phi_calc = IntegratedInformationCalculator()

    def compute_social_phi(self,
                          agent_states: List[AgentState],
                          coupling_matrix: np.ndarray) -> float:
        """
        Compute Φ_social using the coupling matrix between agents.
        Coupling matrix represents information flow/influence between agents.
        """
        # Φ_social is the integration of the multi-agent graph
        # For P6, we use the coupling matrix as the connectivity matrix
        return self.phi_calc.compute_phi(coupling_matrix, np.zeros(len(agent_states)))

    def compute_democratic_volition(self,
                                   group_action: np.ndarray,
                                   individual_preferred_actions: List[np.ndarray]) -> float:
        """
        Quantifies how much the group action reflects individual preferences.
        DV = Mean(CosineSimilarity(GroupAction, IndividualPreference))
        """
        if not individual_preferred_actions:
            return 0.0

        group_norm = np.linalg.norm(group_action)
        if group_norm == 0: return 0.0

        # Optimized: Vectorized cosine similarity computation
        prefs = np.array(individual_preferred_actions)
        pref_norms = np.linalg.norm(prefs, axis=1)
        denom = group_norm * pref_norms
        valid = denom > 1e-12

        dots = np.dot(prefs, group_action)
        similarities = np.zeros(len(prefs))
        similarities[valid] = dots[valid] / denom[valid]

        return float(np.mean(similarities))

    def detect_synergy(self,
                       collective_fwi: float,
                       individual_fwis: List[float]) -> Dict:
        """
        Quantifies the synergy of the group agency.
        Synergy is present if collective FWI outperforms the average individual.
        """
        avg_individual = np.mean(individual_fwis)
        gain = collective_fwi - avg_individual

        return {
            'has_synergy': gain > 0,
            'synergy_gain': gain,
            'super_additivity_ratio': collective_fwi / avg_individual if avg_individual > 0 else 0
        }

class SwarmSimulator:
    """
    Simulates a swarm of agents and their collective volitional dynamics.
    Uses JAX for efficient coupling simulation.
    """

    def __init__(self, n_agents: int = 100):
        self.n_agents = n_agents
        self.fwi_calc = FreeWillIndex()
        self.social_calc = CollectiveFreeWill(self.fwi_calc)

    @staticmethod
    @jax.jit
    def _simulate_coupling_jax(preferences, coupling_matrix):
        """
        Simulates how agent preferences shift due to social coupling.
        new_pref = (1 - strength) * pref + strength * avg_neighbor_pref
        """
        # preferences: (N, 3)
        # coupling_matrix: (N, N)
        # Normalize coupling matrix rows
        row_sums = jnp.sum(coupling_matrix, axis=1, keepdims=True)
        norm_coupling = jnp.where(row_sums > 0, coupling_matrix / row_sums, 0.0)

        neighbor_prefs = jnp.matmul(norm_coupling, preferences)

        # We'll use a fixed internal strength of 0.5 for the JAX part
        return 0.5 * preferences + 0.5 * neighbor_prefs

    def run_step(self,
                 coupling_strength: float = 0.5,
                 noise_level: float = 0.1) -> Dict:
        """
        Simulate one step of swarm interaction and compute collective agency.
        """
        # Generate random agent states for simulation
        agent_states = []
        individual_fwis = []
        preferred_actions = []

        # Simple dynamics for demo
        def dynamics(s, a):
            # Optimized & Robust to batching
            res = s * 0.9
            if a.ndim == 1:
                n = min(len(s), len(a))
                res[:n] += 0.1 * a[:n]
            else:
                if res.ndim == 1:
                    res = np.tile(res, (len(a), 1))
                n = min(res.shape[-1], a.shape[-1])
                res[:, :n] += 0.1 * a[:, :n]
            return res
        bounds = np.ones(3) * 2.0

        # Common goal component for the swarm
        common_goal = np.random.randn(3)

        for i in range(self.n_agents):
            # Individual preference biased towards common goal
            pref_bias = common_goal + np.random.randn(3) * 0.5

            agent = AgentState(
                belief_state=np.random.randn(10),
                goal_state=np.random.rand(5),
                meta_belief=np.random.randn(8) * 0.5,
                action_repertoire=np.random.randn(20, 3) + pref_bias # Bias repertoire
            )
            agent_states.append(agent)

            # Compute individual FWI (simplified for speed in swarm)
            # Use small connectivity for individuals
            conn = np.eye(10) * 0.5
            res = self.fwi_calc.compute(agent, dynamics, conn, bounds)
            individual_fwis.append(res['fwi'])

            # Random preferred action from repertoire
            idx = np.random.randint(0, len(agent.action_repertoire))
            preferred_actions.append(agent.action_repertoire[idx])

        # Build coupling matrix (Structured / Clustered graph for better integration)
        coupling = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            # Connect to 5 neighbors to form clusters
            for j in range(1, 6):
                neighbor = (i + j) % self.n_agents
                coupling[i, neighbor] = coupling_strength
                coupling[neighbor, i] = coupling_strength

        # Add some long-range random connections
        random_conn = np.random.rand(self.n_agents, self.n_agents)
        coupling[random_conn > 0.98] = coupling_strength

        np.fill_diagonal(coupling, 0)

        # Evolve preferences based on coupling (Social Influence)
        refined_prefs = self._simulate_coupling_jax(jnp.array(preferred_actions), jnp.array(coupling))
        refined_prefs = np.array(refined_prefs)

        # Compute Collective Metrics
        social_phi = self.social_calc.compute_social_phi(agent_states, coupling)

        # Final Group Action: Result of social negotiation
        group_action = np.mean(refined_prefs, axis=0)

        democratic_volition = self.social_calc.compute_democratic_volition(group_action, preferred_actions)

        # Distinguish Herd Behavior vs Coordinated Volition
        # High Φ + Low DV = Herd (Internal coherence but lost individual preference)
        # High Φ + High DV = Coordinated Volition (Emergent synergy)

        is_herd = social_phi > 0.7 and democratic_volition < 0.3
        is_coordinated = social_phi > 0.5 and democratic_volition > 0.5

        # Collective FWI (Heuristic for P6)
        collective_fwi = 0.5 * social_phi + 0.5 * democratic_volition

        synergy_report = self.social_calc.detect_synergy(collective_fwi, individual_fwis)

        return {
            'collective_fwi': collective_fwi,
            'social_phi': social_phi,
            'democratic_volition': democratic_volition,
            'individual_fwi_mean': np.mean(individual_fwis),
            'synergy': synergy_report['has_synergy'],
            'synergy_gain': synergy_report['synergy_gain'],
            'status': "COORDINATED" if is_coordinated else ("HERD" if is_herd else "FRAGMENTED")
        }

if __name__ == "__main__":
    print("="*80)
    print("SOCIAL VOLITION SIMULATION (P6) - PHASE TRANSITIONS")
    print("="*80)

    simulator = SwarmSimulator(n_agents=100)

    strengths = [0.1, 0.5, 0.9, 2.0]

    for s in strengths:
        print(f"\n>>> Coupling Strength: {s}")
        results = simulator.run_step(coupling_strength=s)

        print(f"   Collective FWI:      {results['collective_fwi']:.4f}")
        print(f"   Social Integration:  {results['social_phi']:.4f}")
        print(f"   Democratic Volition: {results['democratic_volition']:.4f}")
        print(f"   Individual FWI Mean: {results['individual_fwi_mean']:.4f}")
        print(f"   Status:              {results['status']}")
        print(f"   Synergy Detected:    {results['synergy']}")
