import time
import numpy as np
from free_will_framework import IntegratedInformationCalculator

calc = IntegratedInformationCalculator()
n = 100
conn = np.random.rand(n, n)
conn = (conn + conn.T) / 2
states = [np.random.randn(n) for _ in range(1000)]

# Warmup
calc.compute_phi(conn, states[0])

t0 = time.time()
for s in states:
    calc.compute_phi(conn, s)
t1 = time.time()
print(f"Time for 1000 phi computations (shared connectivity, unique states): {t1 - t0:.4f}s")
