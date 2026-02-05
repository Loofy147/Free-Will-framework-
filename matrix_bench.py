import time
import numpy as np
import jax.numpy as jnp
import jax

@jax.jit
def old_method(connectivity_matrix):
    degree = jnp.sum(connectivity_matrix, axis=1)
    d_inv_sqrt = jnp.where(degree > 0, 1.0 / jnp.sqrt(degree), 0.0)
    D_inv_sqrt = jnp.diag(d_inv_sqrt)
    identity = jnp.eye(connectivity_matrix.shape[0])
    normalized_laplacian = identity - jnp.matmul(jnp.matmul(D_inv_sqrt, connectivity_matrix), D_inv_sqrt)
    return normalized_laplacian

@jax.jit
def new_method(connectivity_matrix):
    degree = jnp.sum(connectivity_matrix, axis=1)
    d_inv_sqrt = jnp.where(degree > 0, 1.0 / jnp.sqrt(degree), 0.0)
    norm_adj = d_inv_sqrt[:, None] * connectivity_matrix * d_inv_sqrt[None, :]
    identity = jnp.eye(connectivity_matrix.shape[0])
    normalized_laplacian = identity - norm_adj
    return normalized_laplacian

n = 100
conn = np.random.rand(n, n)
conn = (conn + conn.T) / 2

# Warmup
old_method(conn).block_until_ready()
new_method(conn).block_until_ready()

t0 = time.time()
for _ in range(100):
    old_method(conn).block_until_ready()
t1 = time.time()
print(f"Old method (100x100, 100 runs): {t1 - t0:.4f}s")

t0 = time.time()
for _ in range(100):
    new_method(conn).block_until_ready()
t1 = time.time()
print(f"New method (100x100, 100 runs): {t1 - t0:.4f}s")
