import sys

with open('integrated_framework.py', 'r') as f:
    content = f.read()

# Update imports
import_search = "from free_will_framework import FreeWillIndex, AgentState, CausalEntropyCalculator, BiologicalSignalSimulator"
import_replace = "from free_will_framework import FreeWillIndex, AgentState, CausalEntropyCalculator, BiologicalSignalSimulator, RealizationManager, RealizationLayer"
content = content.replace(import_search, import_replace)

# Update __init__
init_search = "self.bio_sim = BiologicalSignalSimulator(substrate='Neuromorphic', gain=1.2, noise_sigma=0.03)"
init_replace = init_search + "\n        self.realization_manager = RealizationManager(self.fwi_calc)"
content = content.replace(init_search, init_replace)

# Update compute_full_agency
compute_search = "res = self.fwi_calc.compute(agent, dynamics, conn, bounds)"
compute_replace = "res = self.realization_manager.realize_agency(agent, dynamics, conn, bounds, layer=RealizationLayer.ETHICAL)"
content = content.replace(compute_search, compute_replace)

with open('integrated_framework.py', 'w') as f:
    f.write(content)
