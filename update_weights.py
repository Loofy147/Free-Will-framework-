import sys

with open('/app/free_will_framework.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "self.weights = weights or {" in line:
        new_lines.append(line)
        new_lines.append("            'causal_entropy': 0.0800,\n")
        new_lines.append("            'integration': 0.3000,\n")
        new_lines.append("            'counterfactual': 0.6200,\n")
        new_lines.append("            'metacognition': 0.0000,\n")
        new_lines.append("            'veto_efficacy': 0.0000,\n")
        new_lines.append("            'bayesian_precision': 0.0000,\n")
        new_lines.append("            'constraint_penalty': 0.0000\n")
        new_lines.append("        }\n")
        skip = True
    elif skip and "}" in line:
        skip = False
    elif not skip:
        new_lines.append(line)

with open('/app/free_will_framework.py', 'w') as f:
    f.writelines(new_lines)
