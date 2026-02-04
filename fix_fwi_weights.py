import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

import re

weights_pattern = r'self.weights = weights or \{.*?\}'
new_weights = """self.weights = weights or {
            'causal_entropy': 0.1000,
            'integration': 0.2000,
            'counterfactual': 0.3000,
            'metacognition': 0.0500,
            'veto_efficacy': 0.0500,
            'bayesian_precision': 0.0500,
            'persistence': 0.1000,
            'volitional_integrity': 0.1000,
            'moral_alignment': 0.0500,
            'constraint_penalty': 0.0000
        }"""

content = re.sub(weights_pattern, new_weights, content, flags=re.DOTALL)

with open('free_will_framework.py', 'w') as f:
    f.write(content)
