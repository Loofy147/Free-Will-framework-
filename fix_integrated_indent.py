import sys

with open('integrated_framework.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "        def simulate_collective_volition(self," in line:
        new_lines.append(line.replace("        def simulate_collective_volition", "    def simulate_collective_volition"))
        continue
    new_lines.append(line)

with open('integrated_framework.py', 'w') as f:
    f.writelines(new_lines)
