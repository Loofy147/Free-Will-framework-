import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

import re

# Find the compute method
method_pattern = r'def compute\(self,.*?return \{.*?\}'
# I'll just replace the whole compute method to be safe
new_method = """def compute(self,
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
        }"""

content = re.sub(method_pattern, new_method, content, flags=re.DOTALL)

with open('free_will_framework.py', 'w') as f:
    f.write(content)
