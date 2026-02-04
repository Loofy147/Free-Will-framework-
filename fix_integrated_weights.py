import sys

with open('integrated_framework.py', 'r') as f:
    content = f.read()

weights_search = """        self.fwi_calc.weights = {
            'causal_entropy': 0.10,
            'integration': 0.30,
            'counterfactual': 0.40,
            'metacognition': 0.05,
            'veto_efficacy': 0.05,
            'bayesian_precision': 0.05,
            'persistence': 0.05,
            'volitional_integrity': 0.05,
            'constraint_penalty': 0.0
        }"""

weights_replace = """        self.fwi_calc.weights = {
            'causal_entropy': 0.10,
            'integration': 0.20,
            'counterfactual': 0.30,
            'metacognition': 0.05,
            'veto_efficacy': 0.05,
            'bayesian_precision': 0.05,
            'persistence': 0.10,
            'volitional_integrity': 0.10,
            'moral_alignment': 0.05,
            'constraint_penalty': 0.00
        }"""

content = content.replace(weights_search, weights_replace)

with open('integrated_framework.py', 'w') as f:
    f.write(content)
