import sys

with open('integrated_framework.py', 'r') as f:
    content = f.read()

# Update compute_full_agency to include energy cost
compute_search = "res['biological_signals'] = bold"
compute_replace = compute_search + "\n        res['energy_profile'] = self.bio_sim.compute_energy_cost(res)"
content = content.replace(compute_search, compute_replace)

with open('integrated_framework.py', 'w') as f:
    f.write(content)
