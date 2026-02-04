import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

firewall_class = """
class VolitionalFirewall:
    \"\"\"
    P9: Volitional Integrity (Adversarial Robustness)
    Detects external 'hijacking' of the agent's goal state.
    Monitors the stability of internal motivations against adversarial perturbations.
    \"\"\"
    def __init__(self, history_size: int = 10, threshold: float = 0.5):
        self.goal_history = []
        self.history_size = history_size
        self.threshold = threshold

    def evaluate_integrity(self, current_goal: np.ndarray) -> float:
        \"\"\"
        Calculates a 'hijack_risk' score [0, 1].
        Risk increases if the goal state shifts abruptly or inconsistently.
        \"\"\"
        if not self.goal_history:
            self.goal_history.append(current_goal.copy())
            return 0.0

        # Calculate average similarity to past goals
        similarities = []
        for past_goal in self.goal_history:
            if np.linalg.norm(current_goal) == 0 or np.linalg.norm(past_goal) == 0:
                similarities.append(1.0)
            else:
                sim = np.dot(current_goal, past_goal) / (np.linalg.norm(current_goal) * np.linalg.norm(past_goal) + 1e-9)
                similarities.append(sim)

        avg_similarity = np.mean(similarities)
        hijack_risk = 1.0 - np.clip(avg_similarity, 0, 1)

        # Update history
        self.goal_history.append(current_goal.copy())
        if len(self.goal_history) > self.history_size:
            self.goal_history.pop(0)

        return float(hijack_risk)
"""

search_str = "class FreeWillIndex:"
new_content = content.replace(search_str, firewall_class + "\n" + search_str)

with open('free_will_framework.py', 'w') as f:
    f.write(new_content)
