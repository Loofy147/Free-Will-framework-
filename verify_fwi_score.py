import numpy as np
from adaptive_fwi import AdaptiveFWI, simulate_episode

optimizer = AdaptiveFWI()
optimizer.weights = {'causal_entropy': 0.0800, 'integration': 0.3000, 'counterfactual': 0.6200, 'metacognition': 0.0, 'constraint_penalty': 0.0}

scores = []
for i in range(100):
    episode = simulate_episode(seed=i+5000)
    if episode.emergence_label:
        X = np.array([episode.components[k] for k in optimizer.COMPONENT_KEYS])
        sign = np.array([1, 1, 1, 1, -1], dtype=float)
        fwi = np.clip(X @ (np.array([optimizer.weights[k] for k in optimizer.COMPONENT_KEYS]) * sign), 0, 1)
        scores.append(fwi)

print(f"Mean FWI for emerged agents: {np.mean(scores)}")
print(f"Max FWI: {np.max(scores)}")
