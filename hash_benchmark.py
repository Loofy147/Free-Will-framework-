import time
import numpy as np

state = np.random.randn(10)
n = 100000

t0 = time.time()
for _ in range(n):
    h = state.round(2).tobytes()
t1 = time.time()
print(f"round(2).tobytes(): {t1 - t0:.4f}s")

t0 = time.time()
for _ in range(n):
    h = (state * 100).astype(np.int32).tobytes()
t1 = time.time()
print(f"(state*100).astype(int32).tobytes(): {t1 - t0:.4f}s")
