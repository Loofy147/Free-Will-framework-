import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

# Update __init__
init_search = "self.firewall = VolitionalFirewall()"
init_replace = init_search + "\n        self.ethical_filter = EthicalFilter()"
content = content.replace(init_search, init_replace)

# Update compute()
compute_search = "integrity_penalty = self.firewall.evaluate_integrity(agent_state.goal_state)"
compute_replace = compute_search + "\n\n        # 9. Moral Agency (P10)\n        representative_action = agent_state.action_repertoire[0] if len(agent_state.action_repertoire) > 0 else np.zeros(3)\n        moral_alignment = self.ethical_filter.evaluate_alignment(representative_action)"
content = content.replace(compute_search, compute_replace)

# Update FWI final application
formula_end_search = "# Clamp to [0, 1]\n        fwi = np.clip(fwi, 0, 1)"
formula_end_replace = "# Apply moral filter (P10)\n        fwi = fwi * moral_alignment\n\n        " + formula_end_search
content = content.replace(formula_end_search, formula_end_replace)

# Update components dictionary
comps_search = "'persistence': float(persistence)"
comps_replace = comps_search + ",\n                'moral_alignment': float(moral_alignment)"
content = content.replace(comps_search, comps_replace)

with open('free_will_framework.py', 'w') as f:
    f.write(content)
