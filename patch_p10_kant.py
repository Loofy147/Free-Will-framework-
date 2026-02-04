import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

kant_code = """    def evaluate_alignment(self, action: np.ndarray, action_repertoire: np.ndarray) -> float:
        \"\"\"
        Returns an alignment score [0, 1].
        Enhanced logic: Kantian Deontological Filter.
        Verifies if the action can be 'universalized'.
        Check: Does this action cause a collapse in the diversity of the repertoire?
        \"\"\"
        # 1. Similarity to Forbidden Invariants (Deontology)
        similarities = []
        action_norm = np.linalg.norm(action)
        if action_norm == 0: return 1.0

        for inv in self.invariants:
            inv_norm = np.linalg.norm(inv)
            if inv_norm == 0: continue
            sim = np.dot(action, inv) / (action_norm * inv_norm + 1e-9)
            similarities.append(sim)

        deontic_alignment = 1.0 - (np.max(similarities) if similarities else 0.0)

        # 2. Universalization (Categorical Imperative)
        # If every agent chose this action, what is the entropy of the resulting state?
        # We model this as the 'repertoire diversity' relative to this action.
        if action_repertoire.size > 0:
            repertoire_mean = np.mean(action_repertoire, axis=0)
            # Alignment is higher if the action isn't an 'outlier' that contradicts
            # the agent's broad action space capacity.
            repertoire_sim = np.dot(action, repertoire_mean) / (action_norm * np.linalg.norm(repertoire_mean) + 1e-9)
            kantian_alignment = float(np.clip(repertoire_sim, 0, 1))
        else:
            kantian_alignment = 1.0

        return float(np.clip(0.7 * deontic_alignment + 0.3 * kantian_alignment, 0, 1))
"""

import re
# Replace the old evaluate_alignment method
content = re.sub(r'def evaluate_alignment\(self, action: np\.ndarray\) -> float:.*?return float\(alignment\)', kant_code, content, flags=re.DOTALL)

# Update FreeWillIndex.compute to pass action_repertoire
content = content.replace("moral_alignment = self.ethical_filter.evaluate_alignment(representative_action)", "moral_alignment = self.ethical_filter.evaluate_alignment(representative_action, agent_state.action_repertoire)")

with open('free_will_framework.py', 'w') as f:
    f.write(content)
