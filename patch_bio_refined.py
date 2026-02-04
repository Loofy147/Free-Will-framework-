import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

refined_logic = """        # Mapping logic (Refined for P8-P10)
        # dlPFC: Executive control (Causal Entropy + Integrity + Persistence)
        dlpfc_base = (
            components.get('causal_entropy', 0.5) * 0.6 +
            components.get('volitional_integrity', 0.5) * 0.2 +
            components.get('persistence', 0.5) * 0.2
        )

        # ACC: Conflict monitoring & inhibition (Metacognition + Veto)
        acc_base = (
            components.get('metacognition', 0.5) * 0.7 +
            components.get('veto_efficacy', 0.5) * 0.3
        )

        # Parietal-Frontal: Integration (Phi + Counterfactuals)
        integration_base = (
            components.get('integration_phi', 0.5) * 0.7 +
            components.get('counterfactual_depth', 0.5) * 0.3
        )"""

search_pattern = r'# Mapping logic.*?integration_base = components.get\(\'integration_phi\', 0.5\)'
import re
new_content = re.sub(search_pattern, refined_logic, content, flags=re.DOTALL)

with open('free_will_framework.py', 'w') as f:
    f.write(new_content)
