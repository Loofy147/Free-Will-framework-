"""
ADAPTIVE FREE WILL INDEX — WEIGHT OPTIMIZATION ENGINE
Replaces hand-tuned weights with learned optimal weights via projected gradient descent.
Imports live from free_will_framework.py.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import time

# ---------------------------------------------------------------------------
# Re-use existing project classes
# ---------------------------------------------------------------------------
from free_will_framework import (
    AgentState, FreeWillIndex, EmergenceProof,
    CausalEntropyCalculator, IntegratedInformationCalculator,
    CounterfactualDepthCalculator
)


# ---------------------------------------------------------------------------
# Episode data collector
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """One simulated agent episode with ground-truth labels"""
    components: Dict[str, float]          # CE, Phi, CD, MA, EC (all normalized)
    emergence_label: bool                  # Did emergence proof pass?
    fwi_target: float                     # Target FWI (1.0 if emerged, 0.0 if not)


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project vector onto probability simplex: Σv_i = 1, v_i ≥ 0
    Algorithm: Duchi et al. (2008) — O(n log n)
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cumsum = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cumsum - 1))[0][-1]
    theta = (cumsum[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def _make_dynamics(state_dim: int):
    """Nonlinear dynamics for episode generation"""
    A = np.random.randn(state_dim, state_dim) * 0.3
    A = A / (np.max(np.abs(np.linalg.eigvals(A))) + 0.01)  # stable
    def dynamics(s, a):
        a_flat = a.flatten()
        a_proj = a_flat[:len(s)] if len(a_flat) >= len(s) else np.pad(
            a_flat, (0, len(s) - len(a_flat)))
        return np.tanh(A @ s + 0.15 * a_proj) + np.random.randn(len(s)) * 0.005
    return dynamics


def simulate_episode(seed: int, state_dim: int = 10, n_actions: int = 20) -> Episode:
    """
    Generate one complete episode: create agent → compute components → label.
    Each episode is independent (different random agent).
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    # --- agent with controlled variance ---
    meta_var = rng.uniform(0.1, 2.0)          # controls metacognition
    action_spread = rng.uniform(0.5, 4.0)     # controls counterfactual depth
    connectivity_density = rng.uniform(0.1, 0.9)

    agent = AgentState(
        belief_state=rng.randn(state_dim),
        goal_state=rng.rand(5),
        meta_belief=rng.randn(8) * np.sqrt(meta_var),
        action_repertoire=rng.randn(n_actions, 3) * action_spread
    )

    dynamics = _make_dynamics(state_dim)
    bounds = np.array([2.0, 2.0, 2.0])

    # --- connectivity ---
    conn = rng.rand(state_dim, state_dim) * connectivity_density
    conn = (conn + conn.T) / 2
    np.fill_diagonal(conn, 0)

    # --- raw component scores (replicating FreeWillIndex internals) ---
    ce_calc = CausalEntropyCalculator(time_horizon=20)   # reduced horizon for speed
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

    components = {
        'causal_entropy':    ce_norm,
        'integration':       phi,
        'counterfactual':    cd_norm,
        'metacognition':     ma,
        'constraint_penalty': ec
    }

    # --- emergence label via existing proof logic ---
    # Build a mock fwi_result to feed EmergenceProof
    fwi_result = {
        'fwi': 0.0,  # placeholder
        'counterfactual_count': n_cf,
        'components': {
            'integration_phi': phi
        }
    }
    # self_prediction_accuracy inversely related to meta_var
    self_pred_acc = 1.0 - min(meta_var_actual * 0.15, 0.98)
    proof = EmergenceProof.prove_emergence(agent, fwi_result, self_pred_acc)
    emerged = proof['emergence_proven']

    return Episode(
        components=components,
        emergence_label=emerged,
        fwi_target=1.0 if emerged else 0.0
    )


# ---------------------------------------------------------------------------
# Adaptive optimizer
# ---------------------------------------------------------------------------

@dataclass
class OptimizationTrace:
    """Full convergence record"""
    epoch:          int
    weights:        np.ndarray
    loss:           float
    accuracy:       float   # fraction of episodes where sign(FWI-0.5) == label


class AdaptiveFWI(FreeWillIndex):
    """
    FreeWillIndex with learned weights.

    Optimization objective:
        L = (1/N) Σ [ (FWI_i - target_i)² ]   subject to  w ∈ Δ⁴  (simplex)

    where target_i = 1 if emergence_proven else 0.
    Gradient is analytic (FWI is linear in w) → exact update.
    Simplex projection after every step.
    """

    COMPONENT_KEYS = ['causal_entropy', 'integration', 'counterfactual',
                      'metacognition', 'constraint_penalty']

    def __init__(self):
        super().__init__()                        # start with hand-tuned defaults
        self._trace: List[OptimizationTrace]    = []
        self._optimal_weights: Dict[str, float] = dict(self.weights)  # best so far
        self._best_loss: float                  = float('inf')

    # --- public API --------------------------------------------------------

    def optimize(self, n_episodes: int = 500, lr: float = 0.08,
                 n_epochs: int = 150, verbose: bool = True) -> Dict:
        """
        Run full optimization loop.

        Returns:
            {
                'optimal_weights': {...},
                'convergence':     [OptimizationTrace, ...],
                'final_loss':      float,
                'final_accuracy':  float,
                'episodes_used':   int
            }
        """
        t0 = time.time()
        if verbose:
            print("\n" + "=" * 70)
            print(" ADAPTIVE FWI — WEIGHT OPTIMIZATION")
            print("=" * 70)
            print(f"  Episodes : {n_episodes}")
            print(f"  Epochs   : {n_epochs}")
            print(f"  LR       : {lr}")
            print("-" * 70)

        # 1. collect episodes
        if verbose:
            print("[1/3] Simulating episodes...", flush=True)
        episodes = []
        for i in range(n_episodes):
            episodes.append(simulate_episode(seed=i + 1000))
        n_emerged   = sum(1 for e in episodes if e.emergence_label)
        n_no_emerge = n_episodes - n_emerged
        if verbose:
            print(f"       Emerged: {n_emerged} | Not emerged: {n_no_emerge}")

        # 2. build data matrices
        # X: (N, 5)  — component scores
        # y: (N,)    — target (1 or 0)
        # sign_vec: last column is subtracted not added → multiply by -1
        X = np.array([[e.components[k] for k in self.COMPONENT_KEYS] for e in episodes])
        y = np.array([e.fwi_target for e in episodes])
        sign = np.array([1, 1, 1, 1, -1], dtype=float)   # EC is penalty

        # 3. gradient descent on simplex
        if verbose:
            print("[2/3] Running projected gradient descent...", flush=True)

        w = np.array([self.weights[k] for k in self.COMPONENT_KEYS])  # init = hand-tuned

        for epoch in range(n_epochs):
            # forward: FWI_i = clip(X_i · (w * sign), 0, 1)
            fwi_pred = np.clip(X @ (w * sign), 0.0, 1.0)

            # MSE loss
            residuals = fwi_pred - y
            loss = float(np.mean(residuals ** 2))

            # accuracy
            accuracy = float(np.mean(
                ((fwi_pred >= 0.5) == (y >= 0.5)).astype(float)
            ))

            # analytic gradient  ∂L/∂w_j = (2/N) Σ residual_i * sign_j * X_ij
            grad = (2.0 / len(y)) * (X * sign).T @ residuals

            # update + project
            w = w - lr * grad
            w = _project_simplex(w)

            self._trace.append(OptimizationTrace(
                epoch=epoch, weights=w.copy(), loss=loss, accuracy=accuracy
            ))

            if loss < self._best_loss:
                self._best_loss = loss
                self._optimal_weights = {
                    k: float(w[i]) for i, k in enumerate(self.COMPONENT_KEYS)
                }

            if verbose and (epoch % 30 == 0 or epoch == n_epochs - 1):
                print(f"       Epoch {epoch:>4d} | Loss {loss:.6f} | Acc {accuracy:.3f} | "
                      f"w={np.round(w, 3)}")

        # 4. commit optimal weights
        self.weights = dict(self._optimal_weights)
        elapsed = time.time() - t0

        if verbose:
            print("-" * 70)
            print("[3/3] CONVERGED")
            print(f"       Best loss     : {self._best_loss:.6f}")
            print(f"       Optimal weights: {self._optimal_weights}")
            print(f"       Wall time     : {elapsed:.2f}s")
            print("=" * 70)

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

    def get_convergence_curve(self) -> Dict[str, List]:
        """Return losses and accuracies indexed by epoch for plotting"""
        return {
            'epochs':     [t.epoch     for t in self._trace],
            'losses':     [t.loss      for t in self._trace],
            'accuracies': [t.accuracy  for t in self._trace]
        }

    # --- ablation ----------------------------------------------------------

    def ablation_study(self, episodes: List[Episode] = None) -> Dict[str, Dict]:
        """
        Zero each weight one at a time; measure loss delta.
        Tells you which component contributes most.
        """
        if episodes is None:
            episodes = [simulate_episode(seed=i + 9000) for i in range(100)]

        X = np.array([[e.components[k] for k in self.COMPONENT_KEYS] for e in episodes])
        y = np.array([e.fwi_target for e in episodes])
        sign = np.array([1, 1, 1, 1, -1], dtype=float)

        # baseline with optimal weights
        w_opt = np.array([self._optimal_weights[k] for k in self.COMPONENT_KEYS])
        fwi_base = np.clip(X @ (w_opt * sign), 0, 1)
        loss_base = float(np.mean((fwi_base - y) ** 2))

        results = {}
        for i, key in enumerate(self.COMPONENT_KEYS):
            w_ablated = w_opt.copy()
            w_ablated[i] = 0.0
            # re-project (redistribute removed weight)
            w_ablated = _project_simplex(w_ablated)
            fwi_abl  = np.clip(X @ (w_ablated * sign), 0, 1)
            loss_abl = float(np.mean((fwi_abl - y) ** 2))

            results[key] = {
                'weight_optimal':  float(w_opt[i]),
                'loss_with':       loss_base,
                'loss_without':    loss_abl,
                'delta_loss':      loss_abl - loss_base,
                'importance_rank': 0  # filled below
            }

        # rank by delta (higher delta = more important)
        sorted_keys = sorted(results.keys(), key=lambda k: results[k]['delta_loss'], reverse=True)
        for rank, key in enumerate(sorted_keys, 1):
            results[key]['importance_rank'] = rank

        return results


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    optimizer = AdaptiveFWI()
    report = optimizer.optimize(n_episodes=1000, n_epochs=150, lr=0.1)

    print("\n--- ABLATION STUDY ---")
    ablation = optimizer.ablation_study()
    print(f"{'Component':<22} {'Opt Weight':>10} {'Loss w/':>10} {'Loss w/o':>10} {'ΔLoss':>10} {'Rank':>5}")
    print("-" * 72)
    for key in sorted(ablation, key=lambda k: ablation[k]['importance_rank']):
        a = ablation[key]
        print(f"{key:<22} {a['weight_optimal']:>10.4f} {a['loss_with']:>10.6f} "
              f"{a['loss_without']:>10.6f} {a['delta_loss']:>+10.6f} {a['importance_rank']:>5}")
