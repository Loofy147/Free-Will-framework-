import time
import numpy as np
from adaptive_fwi import simulate_episode

start_time = time.time()
for i in range(50):
    simulate_episode(seed=i)
end_time = time.time()
print(f"Time for 50 episodes: {end_time - start_time:.4f}s")
