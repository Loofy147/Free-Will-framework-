import json
import numpy as np
import pandas as pd
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

def simulate_episode(seed: int = 42, eeg_data: np.ndarray = None, mri_data: Dict = None) -> Episode:
    np.random.seed(seed)

    # Use EEG data as noise source
    noise_gain = 1.0
    if eeg_data is not None:
        eeg_norm = (eeg_data - np.mean(eeg_data)) / (np.std(eeg_data) + 1e-6)
        noise_gain = 1.0 + 0.1 * np.mean(np.abs(eeg_norm))

    # Use MRI data to ground goal and persistence
    cdr_penalty = 0.0
    persistence_bias = 0.5
    if mri_data is not None:
        cdr_penalty = mri_data.get('CDR', 0.0) * 0.5
        nwbv = mri_data.get('nWBV', 0.7)
        persistence_bias = nwbv # Higher volume -> higher persistence potential

    agent = AgentState(
        belief_state=np.random.randn(10) * noise_gain,
        goal_state=np.random.rand(5) * (1.0 - cdr_penalty),
        meta_belief=np.random.randn(8) * 0.5,
        action_repertoire=np.random.randn(20, 3)
    )

    def dynamics(s, a):
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
    persistence = float(np.clip(persistence * persistence_bias, 0, 1))

    veto_calc = VetoMechanism()
    vetoes = 0
    for _ in range(5):
        idx = np.random.randint(0, len(agent.action_repertoire))
        if veto_calc.evaluate_veto(agent.action_repertoire[idx], agent.belief_state, agent.goal_state, dynamics):
            vetoes += 1
    veto_eff = 1.0 - (vetoes / 5.0)

    firewall = VolitionalFirewall()
    firewall.evaluate_integrity(agent.goal_state, agent.meta_belief)
    # Simulate a goal shift based on CDR
    shifted_goal = agent.goal_state * (1.0 - cdr_penalty * 0.5)
    integrity_penalty = firewall.evaluate_integrity(shifted_goal, agent.meta_belief)
    integrity = 1.0 - integrity_penalty

    ethical_filter = EthicalFilter()
    moral_alignment = ethical_filter.evaluate_alignment(agent.action_repertoire[0], agent.action_repertoire)

    components = {
        'causal_entropy':       ce_norm,
        'integration_phi':      phi,
        'counterfactual_depth': cd_norm,
        'metacognition':        ma,
        'veto_efficacy':        veto_eff,
        'bayesian_precision':   0.8,
        'persistence':          persistence,
        'volitional_integrity': integrity,
        'moral_alignment':      moral_alignment,
        'external_constraint':  ec
    }

    # --- ENHANCED emergence label ---
    # Now requires stability and integrity in addition to Phi and CF
    fwi_result = {
        'fwi': 0.0,
        'counterfactual_count': n_cf,
        'components': {
            'integration_phi': phi,
            'persistence': persistence,
            'volitional_integrity': integrity
        }
    }
    self_pred_acc = 1.0 - min(meta_var_actual * 0.15, 0.98)
    proof = EmergenceProof.prove_emergence(agent, fwi_result, self_pred_acc)

    # Enhanced logic: Must have persistence and integrity > 0.4
    emerged = proof['emergence_proven'] and persistence > 0.4 and integrity > 0.4 and cdr_penalty < 0.2

    return Episode(
        components=components,
        emergence_label=emerged,
        fwi_target=1.0 if emerged else 0.0
    )


class AdaptiveFWI(FreeWillIndex):
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
                 n_epochs: int = 150, verbose: bool = True, entropy_reg: float = 0.01) -> Dict:
        t0 = time.time()
        if verbose:
            print("\n" + "=" * 70)
            print(" ADAPTIVE FWI — MULTI-MODAL GROUNDED OPTIMIZATION")
            print("=" * 70)

        # 1. Load EEG data
        try:
            df_eeg = pd.read_csv("data/seizure/Epileptic Seizure Recognition.csv")
            eeg_array = df_eeg.iloc[:, 1:179].values
        except Exception:
            eeg_array = None
            print("[WARN] EEG data not found.")

        # 2. Load MRI data
        try:
            df_mri = pd.read_csv("data/oasis_longitudinal.csv")
            mri_list = df_mri.to_dict('records')
        except Exception:
            mri_list = None
            print("[WARN] MRI data not found.")

        episodes = []
        for i in range(n_episodes):
            eeg_row = eeg_array[i % len(eeg_array)] if eeg_array is not None else None
            mri_row = mri_list[i % len(mri_list)] if mri_list is not None else None
            episodes.append(simulate_episode(seed=i + 1000, eeg_data=eeg_row, mri_data=mri_row))

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

            if entropy_reg > 0:
                grad_entropy = - (1.0 + np.log(w + 1e-9))
                grad += entropy_reg * grad_entropy

            w = w - lr * grad
            w = _project_simplex(w)

            self._trace.append(OptimizationTrace(epoch=epoch, weights=w.copy(), loss=loss, accuracy=accuracy))

            if loss < self._best_loss:
                self._best_loss = loss
                self._optimal_weights = {weight_key_map[k]: float(w[i]) for i, k in enumerate(self.COMPONENT_KEYS)}

            if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
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

@dataclass
class OptimizationTrace:
    epoch: int; weights: np.ndarray; loss: float; accuracy: float

if __name__ == "__main__":
    optimizer = AdaptiveFWI()
    report = optimizer.optimize(n_episodes=2000, n_epochs=500, lr=0.03, entropy_reg=0.04)
    print(f"\nOptimal weights: {report['optimal_weights']}")

    with open("optimized_weights.json", "w") as f:
        json.dump(report["optimal_weights"], f, indent=4)
    print("\n[INFO] Optimized weights saved to optimized_weights.json")
