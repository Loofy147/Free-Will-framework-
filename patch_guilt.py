import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

# Update EthicalFilter to include guilt signal
guilt_code = """    def compute_guilt_signal(self, alignment_score: float, fwi_score: float) -> float:
        \"\"\"
        P10: Guilt Signal
        A precision-weighted prediction error that fires when a high-volition action
        violates a moral constraint.
        Guilt = FWI * (1 - Alignment)
        \"\"\"
        return float(fwi_score * (1.0 - alignment_score))
"""

search_str = "return float(alignment)"
content = content.replace(search_str, search_str + "\n\n" + guilt_code)

# Update FreeWillIndex.compute to include guilt signal
compute_search = "moral_alignment = self.ethical_filter.evaluate_alignment(representative_action)"
compute_replace = compute_search + "\n\n        # 10. Guilt Signal (P10)\n        guilt_signal = self.ethical_filter.compute_guilt_signal(moral_alignment, fwi)"
content = content.replace(compute_search, compute_replace)

# Update components dictionary
comps_search = "'moral_alignment': float(moral_alignment)"
comps_replace = comps_search + ",\n                'guilt_signal': float(guilt_signal)"
content = content.replace(comps_search, comps_replace)

with open('free_will_framework.py', 'w') as f:
    f.write(content)
