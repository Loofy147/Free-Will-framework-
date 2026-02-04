import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

phi_class = """
class IntegratedInformationCalculator:
    \"\"\"
    Φ (Phi) - Integrated Information Theory metric

    Quantifies: Irreducibility of causal structure
    Interpretation: System cannot be decomposed into independent parts

    Enhanced logic: MIP (Minimum Information Partition) proxy using
    Normalized Cut spectral analysis and Mutual Information loss.
    \"\"\"
    def __init__(self):
        self._phi_cache = {}

    @staticmethod
    @jax.jit
    def _compute_spectral_gap_jax(connectivity_matrix):
        \"\"\"Accelerated spectral gap calculation using JAX\"\"\"
        # Build normalized graph Laplacian
        degree = jnp.sum(connectivity_matrix, axis=1)
        # Avoid division by zero
        d_inv_sqrt = jnp.where(degree > 0, 1.0 / jnp.sqrt(degree), 0.0)
        D_inv_sqrt = jnp.diag(d_inv_sqrt)

        identity = jnp.eye(connectivity_matrix.shape[0])
        normalized_laplacian = identity - jnp.matmul(jnp.matmul(D_inv_sqrt, connectivity_matrix), D_inv_sqrt)

        # Compute eigenvalues
        eigenvalues = jnp.linalg.eigvalsh(normalized_laplacian)
        eigenvalues = jnp.sort(eigenvalues)

        # Spectral gap for normalized laplacian = λ_2
        # (λ_1 is always 0 for normalized laplacian)
        return jnp.where(len(eigenvalues) > 1, eigenvalues[1], 0.0)

    def compute_phi(self,
                    connectivity_matrix: np.ndarray,
                    state: np.ndarray) -> float:
        \"\"\"
        Compute Φ - integrated information using MIP proxy logic.
        \"\"\"
        # Cache lookup
        h = hash(connectivity_matrix.tobytes())
        if h in self._phi_cache:
            return self._phi_cache[h]

        n = len(connectivity_matrix)
        if n < 2: return 0.0

        # Logic: Φ is high if the 'cheapest' partition still loses significant information
        # We use the normalized spectral gap of the Laplacian as the MIP proxy
        # Higher λ2 = more integrated / harder to partition without loss

        try:
            phi_spectral = float(self._compute_spectral_gap_jax(connectivity_matrix))
        except:
            # Fallback to numpy
            degree = np.sum(connectivity_matrix, axis=1)
            d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
            D_inv_sqrt = np.diag(d_inv_sqrt)
            identity = np.eye(n)
            norm_lap = identity - D_inv_sqrt @ connectivity_matrix @ D_inv_sqrt
            evs = np.sort(np.linalg.eigvalsh(norm_lap))
            phi_spectral = float(evs[1]) if len(evs) > 1 else 0.0

        # State-dependent modulation: Φ increases with state complexity (entropy)
        # (State with all zeros has lower integration capacity)
        state_entropy = float(np.var(state)) if state.size > 0 else 0.0
        phi = phi_spectral * (1.0 + 0.2 * np.tanh(state_entropy))

        phi = float(np.clip(phi, 0, 1))
        self._phi_cache[h] = phi
        return phi
"""

import re
# Replace the entire old IntegratedInformationCalculator
content = re.sub(r'class IntegratedInformationCalculator:.*?self\._phi_cache\[h\] = phi\n        return phi', phi_class, content, flags=re.DOTALL)

with open('free_will_framework.py', 'w') as f:
    f.write(content)
