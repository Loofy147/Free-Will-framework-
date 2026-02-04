import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

recursive_code = """    def evaluate_integrity(self, current_goal: np.ndarray, meta_belief: np.ndarray) -> float:
        \"\"\"
        Calculates a 'hijack_risk' score [0, 1].
        Enhanced logic: Recursive Integrity Tracking.
        Checks for 'Second-Order Hijacking' by comparing goal stability with
        meta-belief awareness.
        \"\"\"
        if not self.goal_history:
            self.goal_history.append(current_goal.copy())
            return 0.0

        # 1. Goal Stability (First-Order)
        similarities = []
        for past_goal in self.goal_history:
            if np.linalg.norm(current_goal) == 0 or np.linalg.norm(past_goal) == 0:
                similarities.append(1.0)
            else:
                sim = np.dot(current_goal, past_goal) / (np.linalg.norm(current_goal) * np.linalg.norm(past_goal) + 1e-9)
                similarities.append(sim)
        avg_goal_stability = np.mean(similarities)

        # 2. Meta-Awareness (Second-Order)
        # Does the meta_belief 'predict' or 'align' with the goal shift?
        # Higher meta_variance often indicates uncertainty or 'blind spots'
        meta_blindness = np.var(meta_belief) if meta_belief.size > 0 else 1.0

        # Hijack risk is high if goals are unstable AND meta-belief is blind/confused
        hijack_risk = (1.0 - avg_goal_stability) * (1.0 + np.tanh(meta_blindness))
        hijack_risk = float(np.clip(hijack_risk, 0, 1))

        # Update history
        self.goal_history.append(current_goal.copy())
        if len(self.goal_history) > self.history_size:
            self.goal_history.pop(0)

        return hijack_risk
"""

import re
# Replace the old evaluate_integrity method
content = re.sub(r'def evaluate_integrity\(self, current_goal: np\.ndarray\) -> float:.*?return float\(hijack_risk\)', recursive_code, content, flags=re.DOTALL)

# Update FreeWillIndex.compute to pass meta_belief
content = content.replace("integrity_penalty = self.firewall.evaluate_integrity(agent_state.goal_state)", "integrity_penalty = self.firewall.evaluate_integrity(agent_state.goal_state, agent_state.meta_belief)")

with open('free_will_framework.py', 'w') as f:
    f.write(content)
