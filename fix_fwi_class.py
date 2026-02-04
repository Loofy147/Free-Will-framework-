import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

import re

fwi_class_pattern = r'class FreeWillIndex:.*?# ============================================================================\n# PART 3: PROOF OF EMERGENT AGENCY'
new_fwi_class = """class FreeWillIndex:
    \"\"\"
    FWI ∈ [0, 1] - Composite metric quantifying volitional agency

    FWI = w1·CE + w2·Φ + w3·CD + w4·MA + w5·P - w6·EC

    Where:
        CE = Causal Entropy (normalized)
        Φ  = Integrated Information (normalized)
        CD = Counterfactual Depth (normalized)
        MA = Meta-cognitive Awareness (normalized)
        P  = Temporal Persistence (P7)
        EC = External Constraint (penalty term)

    Weights optimized via empirical neuroscience data
    \"\"\"

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Default weights optimized via Machine Learning on biologically-grounded synthetic dataset (P3)
        self.weights = weights or {
            'causal_entropy': 0.1000,
            'integration': 0.3000,
            'counterfactual': 0.4000,
            'metacognition': 0.0500,
            'veto_efficacy': 0.0500,
            'bayesian_precision': 0.0500,
            'persistence': 0.0500,
            'volitional_integrity': 0.0500,
            'constraint_penalty': 0.0000
        }

        self.causal_calc = CausalEntropyCalculator()
        self.phi_calc = IntegratedInformationCalculator()
        self.cf_calc = CounterfactualDepthCalculator()
        self.veto_calc = VetoMechanism()
        self.persistence_calc = TemporalPersistenceCalculator()
        self.firewall = VolitionalFirewall()
        self.ethical_filter = EthicalFilter()
        self.belief_updater = BayesianBeliefUpdater()

    def compute_metacognitive_awareness(self,
                                       agent_state: AgentState,
                                       prediction_error: float) -> float:
        if agent_state.meta_belief.size == 0:
            return 0.0
        meta_variance = np.var(agent_state.meta_belief)
        awareness = np.exp(-meta_variance)
        return float(awareness)

    def compute_external_constraint(self,
                                   agent_state: AgentState,
                                   constitutional_bounds: np.ndarray) -> float:
        n_total = len(agent_state.action_repertoire)
        if n_total == 0: return 1.0
        n_valid = np.sum(
            np.all(np.abs(agent_state.action_repertoire) <= constitutional_bounds,
                   axis=1)
        )
        constraint_ratio = 1 - (n_valid / n_total)
        return float(constraint_ratio)

    def compute(self,
                agent_state: AgentState,
                dynamics_model: callable,
                connectivity_matrix: np.ndarray,
                constitutional_bounds: np.ndarray,
                prediction_error: float = 0.1,
                seed: Optional[int] = None) -> Dict[str, Any]:
        \"\"\"
        Compute Free Will Index with full breakdown
        \"\"\"
        if seed is not None:
            np.random.seed(seed)

        # 1. Causal Entropy
        ce_raw = self.causal_calc.compute_causal_entropy(
            agent_state.belief_state,
            dynamics_model,
            agent_state.action_repertoire
        )
        ce_norm = np.tanh(ce_raw / 10)

        # 2. Integrated Information
        phi = self.phi_calc.compute_phi(connectivity_matrix, agent_state.belief_state)

        # 3. Counterfactual Depth
        n_cf, divergence = self.cf_calc.compute_counterfactual_depth(
            agent_state.belief_state,
            agent_state.action_repertoire,
            dynamics_model
        )
        cd_norm = np.tanh(n_cf / 10)

        # 4. Metacognitive Awareness
        ma = self.compute_metacognitive_awareness(agent_state, prediction_error)

        # 5. Temporal Persistence (P7)
        persistence = self.persistence_calc.compute_persistence(agent_state, dynamics_model)

        # 6. External Constraint (penalty)
        ec = self.compute_external_constraint(agent_state, constitutional_bounds)

        # 7. Veto Efficacy (Free Won't)
        n_veto_samples = 10
        vetoes = 0
        for _ in range(n_veto_samples):
            idx = np.random.randint(0, len(agent_state.action_repertoire))
            action = agent_state.action_repertoire[idx]
            if self.veto_calc.evaluate_veto(action, agent_state.belief_state,
                                           agent_state.goal_state, dynamics_model):
                vetoes += 1
        veto_efficacy = 1 - (vetoes / n_veto_samples)

        # 8. Bayesian Precision
        bayesian_precision = self.belief_updater.precision

        # 9. Volitional Integrity (P9)
        integrity_penalty = self.firewall.evaluate_integrity(agent_state.goal_state)

        # 10. Second-Order Veto (P9)
        is_manipulated = self.firewall.second_order_veto(integrity_penalty, ma)

        # 11. Moral Agency (P10)
        representative_action = agent_state.action_repertoire[0] if len(agent_state.action_repertoire) > 0 else np.zeros(3)
        moral_alignment = self.ethical_filter.evaluate_alignment(representative_action)

        # Compute weighted sum
        fwi_raw = (
            self.weights.get('persistence', 0) * persistence +
            self.weights['causal_entropy'] * ce_norm +
            self.weights['integration'] * phi +
            self.weights['counterfactual'] * cd_norm +
            self.weights['metacognition'] * ma +
            self.weights['veto_efficacy'] * veto_efficacy +
            self.weights['bayesian_precision'] * bayesian_precision -
            self.weights.get('volitional_integrity', 0) * integrity_penalty -
            self.weights['constraint_penalty'] * ec
        )

        # Apply Layered Realizations
        fwi = fwi_raw
        if is_manipulated:
            fwi = fwi * 0.1  # Severe penalty for compromised volition
        fwi = fwi * moral_alignment

        # Clamp to [0, 1]
        fwi = float(np.clip(fwi, 0, 1))

        # 12. Guilt Signal (P10)
        guilt_signal = self.ethical_filter.compute_guilt_signal(moral_alignment, fwi)

        # Interpretation
        if fwi > 0.7:
            interpretation = "HIGH - Strong volitional agency"
        elif fwi > 0.4:
            interpretation = "MODERATE - Limited agency"
        else:
            interpretation = "LOW - Highly constrained/reactive"

        return {
            'fwi': fwi,
            'components': {
                'causal_entropy': float(ce_norm),
                'integration_phi': float(phi),
                'counterfactual_depth': float(cd_norm),
                'metacognition': float(ma),
                'veto_efficacy': float(veto_efficacy),
                'bayesian_precision': float(bayesian_precision),
                'external_constraint': float(ec),
                'persistence': float(persistence),
                'volitional_integrity': float(1.0 - integrity_penalty),
                'moral_alignment': float(moral_alignment),
                'guilt_signal': float(guilt_signal),
                'second_order_veto_active': bool(is_manipulated)
            },
            'interpretation': interpretation,
            'counterfactual_count': int(n_cf),
            'counterfactual_divergence': float(divergence)
        }

# ============================================================================
# PART 3: PROOF OF EMERGENT AGENCY"""

new_content = re.sub(fwi_class_pattern, new_fwi_class, content, flags=re.DOTALL)

with open('free_will_framework.py', 'w') as f:
    f.write(new_content)
