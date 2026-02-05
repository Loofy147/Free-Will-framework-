import time
import numpy as np

n = 100
A = np.random.rand(n, n)
d = np.random.rand(n)
D = np.diag(d)

t0 = time.time()
for _ in range(1000):
    res1 = D @ A @ D
t1 = time.time()
print(f"D @ A @ D: {t1 - t0:.4f}s")

t0 = time.time()
for _ in range(1000):
    res2 = d[:, None] * A * d[None, :]
t1 = time.time()
print(f"d[:, None] * A * d[None, :]: {t1 - t0:.4f}s")

print(f"Results match: {np.allclose(res1, res2)}")
