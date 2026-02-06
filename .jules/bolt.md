# Bolt's Journal
## 2026-02-03 - [Vectorization & Hash Optimization]
**Learning:** np.pad is extremely slow for small arrays in hot loops (~25x slower than manual zeros+slice). Also, tuple(np.round(state, 2)) is a bottleneck in set hashing for reachable states.
**Action:** Replace np.pad with np.zeros + slice. Use state.round(2).tobytes() for faster hashing, and store representative states in a dictionary if needed for further simulation.

## 2026-02-04 - [Matrix Optimization & Caching Bug]
**Learning:** D @ A @ D is 3x slower than d[:, None] * A * d[None, :] even for moderate matrix sizes. Also discovered a critical caching bug in IntegratedInformationCalculator where modulation was lost due to improper cache keying.
**Action:** Use broadcasting for diagonal matrix multiplications. Implement dual-layer caching (cache expensive spectral part, compute cheap modulation) to ensure correctness and performance.
## 2026-02-06 - [Pairwise Distance Optimization]
**Learning:** Using np.linalg.norm with broadcasting for pairwise distances creates large intermediate arrays and is ~10x slower than the dot-product expansion formula (||x-y||^2 = ||x||^2 + ||y||^2 - 2x.y) for moderate number of vectors.
**Action:** Use the expanded dot-product formula for pairwise distances in CounterfactualDepthCalculator and similar components.

## 2026-02-06 - [Vectorization of Cosine Similarity]
**Learning:** Python loops over cosine similarity calculations are a bottleneck in swarm simulations.
**Action:** Vectorize cosine similarity using np.dot(matrix, vector) and axis-wise norm calculations.
