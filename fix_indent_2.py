import sys

with open('free_will_framework.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "        def evaluate_alignment(self," in line:
        new_lines.append(line.replace("        def evaluate_alignment", "    def evaluate_alignment"))
        continue
    new_lines.append(line)

with open('free_will_framework.py', 'w') as f:
    f.writelines(new_lines)
