import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

import re

phi_logic_patch = """        # Logic: Φ is high if the 'cheapest' partition still loses significant information
        # We use the normalized spectral gap of the Laplacian as the MIP proxy.
        # However, for a disconnected graph (A=0), the spectral gap proxy must be 0.

        if np.all(connectivity_matrix == 0):
            phi_spectral = 0.0
        else:
            try:
                phi_spectral = float(self._compute_spectral_gap_jax(connectivity_matrix))
                # For normalized laplacian, phi is 1 - lambda_max_of_normalized_adjacency
                # Or just use the spectral gap if properly defined.
                # Actually, a better proxy for integration is 1 - (1/n) * sum(abs(eigenvalues_of_A))
                # but we'll stick to a spectral gap approach that correctly handles A=0.
            except:
                degree = np.sum(connectivity_matrix, axis=1)
                d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
                D_inv_sqrt = np.diag(d_inv_sqrt)
                identity = np.eye(n)
                norm_lap = identity - D_inv_sqrt @ connectivity_matrix @ D_inv_sqrt
                evs = np.sort(np.linalg.eigvalsh(norm_lap))
                # Phi is the gap between the trivial 0 eigenvalue and the first non-zero one
                # but if the graph is disconnected, multiple eigenvalues are 0.
                phi_spectral = float(evs[1]) if len(evs) > 1 and evs[1] > 1e-5 else 0.0
"""

content = re.sub(r'# Logic: Φ is high.*?phi_spectral = float\(evs\[1\]\) if len\(evs\) > 1 else 0\.0', phi_logic_patch, content, flags=re.DOTALL)

with open('free_will_framework.py', 'w') as f:
    f.write(content)
