import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

# Update __init__
init_search = "self.persistence_calc = TemporalPersistenceCalculator()"
init_replace = init_search + "\n        self.firewall = VolitionalFirewall()"
content = content.replace(init_search, init_replace)

# Update weights
weights_search = "'constraint_penalty': 0.0000"
weights_replace = "'constraint_penalty': 0.0000,\n            'volitional_integrity': 0.0500"
content = content.replace(weights_search, weights_replace)

# Update compute()
compute_search = "bayesian_precision = self.belief_updater.precision"
compute_replace = compute_search + "\n\n        # 8. Volitional Integrity (P9)\n        integrity_penalty = self.firewall.evaluate_integrity(agent_state.goal_state)"
content = content.replace(compute_search, compute_replace)

# Update FWI formula
formula_search = "self.weights['bayesian_precision'] * bayesian_precision -"
formula_replace = formula_search + "\n            self.weights.get('volitional_integrity', 0) * integrity_penalty -"
content = content.replace(formula_search, formula_replace)

# Update components dictionary
comps_search = "'persistence': float(persistence)"
comps_replace = comps_search + ",\n                'volitional_integrity': float(1.0 - integrity_penalty)"
content = content.replace(comps_search, comps_replace)

with open('free_will_framework.py', 'w') as f:
    f.write(content)
