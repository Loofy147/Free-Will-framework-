import sys

with open('free_will_framework.py', 'r') as f:
    lines = f.readlines()

new_lines = []
guilt_line = ""
for line in lines:
    if "guilt_signal = self.ethical_filter.compute_guilt_signal(moral_alignment, fwi)" in line:
        guilt_line = line
        continue
    if "fwi = np.clip(fwi, 0, 1)" in line:
        new_lines.append(line)
        # Add guilt signal after FWI is calculated and clipped
        new_lines.append(guilt_line.replace("fwi", "fwi")) # it was already fwi
        continue
    new_lines.append(line)

with open('free_will_framework.py', 'w') as f:
    f.writelines(new_lines)
