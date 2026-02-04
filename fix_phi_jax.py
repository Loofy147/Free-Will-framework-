import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

import re

phi_jax_patch = """    @staticmethod
    @jax.jit
    def _compute_spectral_gap_jax(connectivity_matrix):
        \"\"\"Accelerated spectral gap calculation using JAX\"\"\"
        # If all zeros, return 0
        all_zeros = jnp.all(connectivity_matrix == 0)

        # Build normalized graph Laplacian
        degree = jnp.sum(connectivity_matrix, axis=1)
        d_inv_sqrt = jnp.where(degree > 0, 1.0 / jnp.sqrt(degree), 0.0)
        D_inv_sqrt = jnp.diag(d_inv_sqrt)

        identity = jnp.eye(connectivity_matrix.shape[0])
        normalized_laplacian = identity - jnp.matmul(jnp.matmul(D_inv_sqrt, connectivity_matrix), D_inv_sqrt)

        # Compute eigenvalues
        eigenvalues = jnp.linalg.eigvalsh(normalized_laplacian)
        eigenvalues = jnp.sort(eigenvalues)

        # Spectral gap for normalized laplacian = λ_2
        # Use a small epsilon to handle numerical noise
        gap = jnp.where(eigenvalues[1] > 1e-5, eigenvalues[1], 0.0)

        return jnp.where(all_zeros, 0.0, gap)"""

content = re.sub(r'@staticmethod.*?return jnp\.where\(len\(eigenvalues\) > 1, eigenvalues\[1\], 0\.0\)', phi_jax_patch, content, flags=re.DOTALL)

with open('free_will_framework.py', 'w') as f:
    f.write(content)
