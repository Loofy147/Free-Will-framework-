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

def simulate_episode(seed: int = 42, eeg_data: np.ndarray = None, mri_transition: Dict = None) -> Episode:
    np.random.seed(seed)

    # Use EEG data as noise source
    noise_gain = 1.0
    if eeg_data is not None:
        eeg_norm = (eeg_data - np.mean(eeg_data)) / (np.std(eeg_data) + 1e-6)
        noise_gain = 1.0 + 0.1 * np.mean(np.abs(eeg_norm))

    # Use MRI transitions to ground persistence (P7) and integrity (P9)
    # Transition dictionary contains {'v1': visit1_data, 'v2': visit2_data}
    persistence_potential = 0.5
    integrity_baseline = 0.5
    moral_bias = 0.5

    if mri_transition is not None:
        v1 = mri_transition['v1']
        v2 = mri_transition['v2']

        # Persistence: Ratio of brain volume retention (nWBV)
        persistence_potential = v2['nWBV'] / (v1['nWBV'] + 1e-9)

        # Integrity: Stability of mental state (CDR)
        # If CDR increases (worsens), integrity decreases
        integrity_baseline = 1.0 - max(0, v2['CDR'] - v1['CDR'])

        # Moral bias: Correlate with cognitive health (MMSE)
        # Higher MMSE (~30) -> potentially higher capacity for complex moral reasoning
        moral_bias = v2['MMSE'] / 30.0 if not np.isnan(v2['MMSE']) else 0.8

    agent = AgentState(
        belief_state=np.random.randn(10) * noise_gain,
        goal_state=np.random.rand(5) * integrity_baseline,
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

    # --- Component scores ---
    ce_calc = CausalEntropyCalculator(time_horizon=20)
    ce_norm = float(np.tanh(ce_calc.compute_causal_entropy(agent.belief_state, dynamics, agent.action_repertoire) / 10))

    phi_calc = IntegratedInformationCalculator()
    phi      = float(phi_calc.compute_phi(conn, agent.belief_state))

    cf_calc  = CounterfactualDepthCalculator()
    n_cf, _  = cf_calc.compute_counterfactual_depth(agent.belief_state, agent.action_repertoire, dynamics)
    cd_norm  = float(np.tanh(n_cf / 10))

    meta_var = float(np.var(agent.meta_belief))
    ma = float(np.exp(-meta_var))

    n_valid  = int(np.sum(np.all(np.abs(agent.action_repertoire) <= bounds, axis=1)))
    ec       = float(1.0 - n_valid / 20.0)

    persistence_calc = TemporalPersistenceCalculator()
    persistence = persistence_calc.compute_persistence(agent, dynamics)
    persistence = float(np.clip(persistence * persistence_potential, 0, 1))

    veto_calc = VetoMechanism()
    veto_eff = 1.0 - (np.sum([veto_calc.evaluate_veto(agent.action_repertoire[np.random.randint(20)], agent.belief_state, agent.goal_state, dynamics) for _ in range(5)]) / 5.0)

    firewall = VolitionalFirewall()
    firewall.evaluate_integrity(agent.goal_state, agent.meta_belief)
    integrity_penalty = firewall.evaluate_integrity(agent.goal_state * 0.9, agent.meta_belief)
    integrity = float(np.clip((1.0 - integrity_penalty) * integrity_baseline, 0, 1))

    ethical_filter = EthicalFilter()
    moral_alignment = float(np.clip(ethical_filter.evaluate_alignment(agent.action_repertoire[0], agent.action_repertoire) * moral_bias, 0, 1))

    components = {
        'causal_entropy': ce_norm, 'integration_phi': phi, 'counterfactual_depth': cd_norm,
        'metacognition': ma, 'veto_efficacy': veto_eff, 'bayesian_precision': 0.85,
        'persistence': persistence, 'volitional_integrity': integrity,
        'moral_alignment': moral_alignment, 'external_constraint': ec
    }

    # Emergence target
    fwi_res = {'fwi': 0.0, 'counterfactual_count': n_cf, 'components': {'integration_phi': phi}}
    proof = EmergenceProof.prove_emergence(agent, fwi_res, 1.0 - min(meta_var*0.1, 0.9))

    # Grounded emergence: high integration, alternatives, AND temporal/ethical stability
    emerged = proof['emergence_proven'] and persistence > 0.4 and integrity > 0.4 and moral_alignment > 0.3

    return Episode(components=components, emergence_label=emerged, fwi_target=1.0 if emerged else 0.0)


class AdaptiveFWI(FreeWillIndex):
    COMPONENT_KEYS = [
        'causal_entropy', 'integration_phi', 'counterfactual_depth',
        'metacognition', 'veto_efficacy', 'bayesian_precision',
        'persistence', 'volitional_integrity', 'moral_alignment',
        'external_constraint'
    ]

    def optimize(self, n_episodes: int = 5000, lr: float = 0.02,
                 n_epochs: int = 1000, verbose: bool = True, entropy_reg: float = 0.05,
                 balance_reg: float = 0.02) -> Dict:
        t0 = time.time()
        if verbose:
            print("\n" + "=" * 70)
            print(" ADAPTIVE FWI — MASSIVE SCALE MULTI-TASK OPTIMIZATION")
            print("=" * 70)

        # 1. Load Grounding Data
        try:
            df_eeg = pd.read_csv("data/seizure/Epileptic Seizure Recognition.csv")
            eeg_array = df_eeg.iloc[:, 1:179].values
        except: eeg_array = None

        try:
            df_mri = pd.read_csv("data/oasis_longitudinal.csv")
            # Create transitions: pair consecutive visits for same subjects
            mri_transitions = []
            for sub_id, group in df_mri.groupby("Subject ID"):
                group = group.sort_values("Visit")
                if len(group) >= 2:
                    rows = group.to_dict('records')
                    for j in range(len(rows)-1):
                        mri_transitions.append({'v1': rows[j], 'v2': rows[j+1]})
        except: mri_transitions = None

        episodes = []
        for i in range(n_episodes):
            eeg_row = eeg_array[i % len(eeg_array)] if eeg_array is not None else None
            mri_trans = mri_transitions[i % len(mri_transitions)] if mri_transitions else None
            episodes.append(simulate_episode(seed=i + 5000, eeg_data=eeg_row, mri_transition=mri_trans))

        X = np.array([[e.components[k] for k in self.COMPONENT_KEYS] for e in episodes])
        y = np.array([e.fwi_target for e in episodes])

        weight_key_map = {
            'causal_entropy': 'causal_entropy', 'integration_phi': 'integration',
            'counterfactual_depth': 'counterfactual', 'metacognition': 'metacognition',
            'veto_efficacy': 'veto_efficacy', 'bayesian_precision': 'bayesian_precision',
            'persistence': 'persistence', 'volitional_integrity': 'volitional_integrity',
            'moral_alignment': 'moral_alignment', 'external_constraint': 'constraint_penalty'
        }

        sign = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, -1], dtype=float)
        w = np.array([self.weights[weight_key_map[k]] for k in self.COMPONENT_KEYS])

        best_loss = float('inf')
        optimal_w = w.copy()

        for epoch in range(n_epochs):
            # Adaptive LR
            current_lr = lr * (0.999 ** epoch)

            fwi_pred = np.clip(X @ (w * sign), 0.0, 1.0)
            residuals = fwi_pred - y
            loss = float(np.mean(residuals ** 2))

            grad = (2.0 / len(y)) * (X * sign).T @ residuals

            if entropy_reg > 0:
                # Maximize entropy to explore
                grad += entropy_reg * (- (1.0 + np.log(w + 1e-9)))

            if balance_reg > 0:
                # Dirichlet-like penalty: - sum log(w) -> grad: - 1/w
                # This pushes weights AWAY from zero
                grad -= balance_reg * (1.0 / (w + 1e-6))

            w = w - current_lr * grad
            w = _project_simplex(w)

            if loss < best_loss:
                best_loss = loss
                optimal_w = w.copy()

            if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
                acc = float(np.mean(((fwi_pred >= 0.5) == (y >= 0.5)).astype(float)))
                print(f"       Epoch {epoch:>4d} | Loss {loss:.6f} | Acc {acc:.3f} | LR {current_lr:.5f}")

        final_weights = {weight_key_map[k]: float(optimal_w[i]) for i, k in enumerate(self.COMPONENT_KEYS)}
        self.weights = final_weights

        return {
            'optimal_weights': final_weights,
            'final_loss': best_loss,
            'wall_time_s': time.time() - t0
        }

if __name__ == "__main__":
    optimizer = AdaptiveFWI()
    # MASSIVE SCALE
    report = optimizer.optimize(n_episodes=5000, n_epochs=1000, lr=0.02, entropy_reg=0.04, balance_reg=0.03)
    print(f"\nOptimal weights: {report['optimal_weights']}")
    with open("optimized_weights.json", "w") as f:
        json.dump(report["optimal_weights"], f, indent=4)
