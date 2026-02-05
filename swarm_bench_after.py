import time
import numpy as np
from social_volition import SwarmSimulator

sim = SwarmSimulator(n_agents=50)
start_time = time.time()
sim.run_step(coupling_strength=0.5)
end_time = time.time()
print(f"Time for SwarmSimulator.run_step (50 agents): {end_time - start_time:.4f}s")
