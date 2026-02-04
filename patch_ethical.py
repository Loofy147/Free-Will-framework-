import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

ethical_class = """
class EthicalFilter:
    \"\"\"
    P10: Moral Agency (Ethical Constraints)
    Bridges volition with responsibility.
    Evaluates actions against core moral invariants (e.g., non-harm, honesty).
    \"\"\"
    def __init__(self, moral_invariants: Optional[List[np.ndarray]] = None):
        # Default invariants: simple vectors in action space representing "forbidden" zones
        self.invariants = moral_invariants or [np.array([1.0, 1.0, 1.0])]

    def evaluate_alignment(self, action: np.ndarray) -> float:
        \"\"\"
        Returns an alignment score [0, 1].
        1.0 = Perfectly aligned with moral invariants.
        0.0 = Violates moral invariants.
        \"\"\"
        # Calculate distance to forbidden invariants
        similarities = []
        action_norm = np.linalg.norm(action)
        if action_norm == 0: return 1.0

        for inv in self.invariants:
            inv_norm = np.linalg.norm(inv)
            if inv_norm == 0: continue
            sim = np.dot(action, inv) / (action_norm * inv_norm + 1e-9)
            similarities.append(sim)

        if not similarities: return 1.0

        max_sim = np.max(similarities)
        # Alignment is inverse of similarity to "forbidden" patterns
        alignment = 1.0 - np.clip(max_sim, 0, 1)
        return float(alignment)
"""

search_str = "class FreeWillIndex:"
new_content = content.replace(search_str, ethical_class + "\n" + search_str)

with open('free_will_framework.py', 'w') as f:
    f.write(new_content)
