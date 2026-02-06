"""
ADAPTIVE FWI - WEIGHT OPTIMIZATION (P1)
Learns optimal FWI weights via analytic gradient descent on simulated datasets.
Optimizes across all 10 realization dimensions.
"""

import numpy as np
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from free_will_framework import (
    AgentState, FreeWillIndex, CausalEntropyCalculator,
    IntegratedInformationCalculator, CounterfactualDepthCalculator,
    EmergenceProof, VetoMechanism, BayesianBeliefUpdater,
    TemporalPersistenceCalculator, VolitionalFirewall, EthicalFilter
)

@dataclass
class Episode:
    """A single volitional trajectory with its component scores and label"""
    components:      Dict[str, float]
    emergence_label: bool
    fwi_target:      float


def _project_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Projection onto the probability simplex: sum(w)=z, w >= 0"""
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n_features + 1) > (cssv - z))[0][-1]
    theta = float(cssv[rho] - z) / (rho + 1)
    return np.maximum(v - theta, 0)


def simulate_episode(seed: int = 42) -> Episode:
    """
    Simulates a single agent state and computes its FWI components.
    Used to generate 'ground truth' for weight optimization.
    """
    np.random.seed(seed)

    agent = AgentState(
        belief_state=np.random.randn(10),
        goal_state=np.random.rand(5),
        meta_belief=np.random.randn(8) * 0.5,
        action_repertoire=np.random.randn(20, 3)
    )

    def dynamics(s, a):
        # Optimized: Avoid np.zeros/np.pad in hot loop (Bolt Journal)
        res = s * 0.9
        if a.ndim == 1:
            n = min(len(s), len(a))
            res[:n] += 0.1 * a[:n]
        else:
            if res.ndim == 1: res = np.tile(res, (len(a), 1))
            n = min(res.shape[-1], a.shape[-1])
            res[:, :n] += 0.1 * a[:, :n]
        return res

    bounds = np.ones(3) * 2.0
    conn = np.random.rand(10, 10)
    conn = (conn + conn.T) / 2
    np.fill_diagonal(conn, 0)

    # --- raw component scores ---
    ce_calc = CausalEntropyCalculator(time_horizon=20)
    ce_raw  = ce_calc.compute_causal_entropy(agent.belief_state, dynamics, agent.action_repertoire)
    ce_norm = float(np.tanh(ce_raw / 10))

    phi_calc = IntegratedInformationCalculator()
    phi      = float(phi_calc.compute_phi(conn, agent.belief_state))

    cf_calc  = CounterfactualDepthCalculator()
    n_cf, _  = cf_calc.compute_counterfactual_depth(agent.belief_state, agent.action_repertoire, dynamics)
    cd_norm  = float(np.tanh(n_cf / 10))

    meta_var_actual = float(np.var(agent.meta_belief))
    ma = float(np.exp(-meta_var_actual))

    n_total  = len(agent.action_repertoire)
    n_valid  = int(np.sum(np.all(np.abs(agent.action_repertoire) <= bounds, axis=1)))
    ec       = float(1.0 - n_valid / max(n_total, 1))

    persistence_calc = TemporalPersistenceCalculator()
    persistence = persistence_calc.compute_persistence(agent, dynamics)

    veto_calc = VetoMechanism()
    vetoes = 0
    for _ in range(5):
        idx = np.random.randint(0, len(agent.action_repertoire))
        if veto_calc.evaluate_veto(agent.action_repertoire[idx], agent.belief_state, agent.goal_state, dynamics):
            vetoes += 1
    veto_eff = 1.0 - (vetoes / 5.0)

    belief_updater = BayesianBeliefUpdater()
    precision = belief_updater.precision

    firewall = VolitionalFirewall()
    integrity_penalty = firewall.evaluate_integrity(agent.goal_state, agent.meta_belief)
    integrity = 1.0 - integrity_penalty

    ethical_filter = EthicalFilter()
    moral_alignment = ethical_filter.evaluate_alignment(agent.action_repertoire[0], agent.action_repertoire)

    components = {
        'causal_entropy':       ce_norm,
        'integration_phi':      phi,
        'counterfactual_depth': cd_norm,
        'metacognition':        ma,
        'veto_efficacy':        veto_eff,
        'bayesian_precision':   precision,
        'persistence':          persistence,
        'volitional_integrity': integrity,
        'moral_alignment':      moral_alignment,
        'external_constraint':  ec
    }

    # --- emergence label via existing proof logic ---
    fwi_result = {
        'fwi': 0.0,
        'counterfactual_count': n_cf,
        'components': {
            'integration_phi': phi
        }
    }
    self_pred_acc = 1.0 - min(meta_var_actual * 0.15, 0.98)
    proof = EmergenceProof.prove_emergence(agent, fwi_result, self_pred_acc)
    emerged = proof['emergence_proven']

    return Episode(
        components=components,
        emergence_label=emerged,
        fwi_target=1.0 if emerged else 0.0
    )


class AdaptiveFWI(FreeWillIndex):
    """
    FreeWillIndex with learned weights across all 10 dimensions.
    """

    COMPONENT_KEYS = [
        'causal_entropy', 'integration_phi', 'counterfactual_depth',
        'metacognition', 'veto_efficacy', 'bayesian_precision',
        'persistence', 'volitional_integrity', 'moral_alignment',
        'external_constraint'
    ]

    def __init__(self):
        super().__init__()
        self._trace: List[OptimizationTrace]    = []
        self._optimal_weights: Dict[str, float] = dict(self.weights)
        self._best_loss: float                  = float('inf')

    def optimize(self, n_episodes: int = 500, lr: float = 0.08,
                 n_epochs: int = 150, verbose: bool = True) -> Dict:
        t0 = time.time()
        if verbose:
            print("\n" + "=" * 70)
            print(" ADAPTIVE FWI — WEIGHT OPTIMIZATION (10 DIMENSIONS)")
            print("=" * 70)

        episodes = []
        for i in range(n_episodes):
            episodes.append(simulate_episode(seed=i + 1000))

        X = np.array([[e.components[k] for k in self.COMPONENT_KEYS] for e in episodes])
        y = np.array([e.fwi_target for e in episodes])

        weight_key_map = {
            'causal_entropy': 'causal_entropy',
            'integration_phi': 'integration',
            'counterfactual_depth': 'counterfactual',
            'metacognition': 'metacognition',
            'veto_efficacy': 'veto_efficacy',
            'bayesian_precision': 'bayesian_precision',
            'persistence': 'persistence',
            'volitional_integrity': 'volitional_integrity',
            'moral_alignment': 'moral_alignment',
            'external_constraint': 'constraint_penalty'
        }

        sign = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, -1], dtype=float)

        w = np.array([self.weights[weight_key_map[k]] for k in self.COMPONENT_KEYS])

        for epoch in range(n_epochs):
            fwi_pred = np.clip(X @ (w * sign), 0.0, 1.0)
            residuals = fwi_pred - y
            loss = float(np.mean(residuals ** 2))
            accuracy = float(np.mean(((fwi_pred >= 0.5) == (y >= 0.5)).astype(float)))
            grad = (2.0 / len(y)) * (X * sign).T @ residuals
            w = w - lr * grad
            w = _project_simplex(w)

            self._trace.append(OptimizationTrace(
                epoch=epoch, weights=w.copy(), loss=loss, accuracy=accuracy
            ))

            if loss < self._best_loss:
                self._best_loss = loss
                self._optimal_weights = {
                    weight_key_map[k]: float(w[i]) for i, k in enumerate(self.COMPONENT_KEYS)
                }

            if verbose and (epoch % 30 == 0 or epoch == n_epochs - 1):
                print(f"       Epoch {epoch:>4d} | Loss {loss:.6f} | Acc {accuracy:.3f}")

        self.weights = dict(self._optimal_weights)
        elapsed = time.time() - t0
        return {
            'optimal_weights':  self._optimal_weights,
            'convergence':      self._trace,
            'final_loss':       self._best_loss,
            'final_accuracy':   self._trace[-1].accuracy if self._trace else 0.0,
            'episodes_used':    n_episodes,
            'wall_time_s':      elapsed
        }

    def get_optimal_weights(self) -> Dict[str, float]:
        return dict(self._optimal_weights)

@dataclass
class OptimizationTrace:
    epoch:          int
    weights:        np.ndarray
    loss:           float
    accuracy:       float

if __name__ == "__main__":
    optimizer = AdaptiveFWI()
    report = optimizer.optimize(n_episodes=200, n_epochs=100, lr=0.1)
    print(f"Optimal weights: {report['optimal_weights']}")
