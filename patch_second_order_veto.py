import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

# Update VolitionalFirewall to include second-order veto
veto_code = """    def second_order_veto(self, hijack_risk: float, metacognition: float) -> bool:
        \"\"\"
        P9: Second-Order Veto
        Vetoes the 'desire' if the hijack risk is high AND metacognition is low
        (indicating the agent isn't aware its goals are being manipulated).
        \"\"\"
        return hijack_risk > self.threshold and metacognition < 0.6
"""

search_str = "return float(hijack_risk)"
content = content.replace(search_str, search_str + "\n\n" + veto_code)

# Update FreeWillIndex.compute to include second-order veto
compute_search = "integrity_penalty = self.firewall.evaluate_integrity(agent_state.goal_state)"
compute_replace = compute_search + "\n\n        # 11. Second-Order Veto (P9)\n        is_manipulated = self.firewall.second_order_veto(integrity_penalty, ma)"
content = content.replace(compute_search, compute_replace)

# Update FWI final application to respect second-order veto
veto_apply_search = "fwi = fwi * moral_alignment"
veto_apply_replace = "if is_manipulated:\n            fwi = fwi * 0.1  # Severe penalty for compromised volition\n        fwi = fwi * moral_alignment"
content = content.replace(veto_apply_search, veto_apply_replace)

# Update components dictionary
comps_search = "'volitional_integrity': float(1.0 - integrity_penalty)"
comps_replace = comps_search + ",\n                'second_order_veto_active': bool(is_manipulated)"
content = content.replace(comps_search, comps_replace)

with open('free_will_framework.py', 'w') as f:
    f.write(content)
