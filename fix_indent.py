import sys

with open('free_will_framework.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "        def evaluate_integrity(self," in line:
        new_lines.append(line.replace("        def evaluate_integrity", "    def evaluate_integrity"))
        continue
    new_lines.append(line)

with open('free_will_framework.py', 'w') as f:
    f.writelines(new_lines)
