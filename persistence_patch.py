import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

persistence_class = """
class TemporalPersistenceCalculator:
    \"\"\"
    P7: Temporal Persistence - Ability to maintain volitional integrity over time.
    Measures the stability of goal-directedness across simulated horizons.
    \"\"\"
    def compute_persistence(self,
                           agent_state: AgentState,
                           dynamics_model: callable,
                           steps: int = 20) -> float:
        current_state = agent_state.belief_state.copy()
        goal = agent_state.goal_state
        norm_goal = np.linalg.norm(goal)
        if norm_goal == 0: return 1.0

        initial_alignment = np.dot(current_state[:len(goal)], goal) / (np.linalg.norm(current_state[:len(goal)]) * norm_goal + 1e-9)

        path_alignments = []
        state = current_state
        for _ in range(steps):
            # Take a random action from repertoire to see if goal is still pursued/reachable
            action = agent_state.action_repertoire[np.random.randint(len(agent_state.action_repertoire))]
            state = dynamics_model(state, action)
            alignment = np.dot(state[:len(goal)], goal) / (np.linalg.norm(state[:len(goal)]) * norm_goal + 1e-9)
            path_alignments.append(alignment)

        persistence = np.mean(path_alignments)
        return float(np.clip(persistence, 0, 1))
"""

search_str = "class CounterfactualDepthCalculator:"
new_content = content.replace(search_str, persistence_class + "\n" + search_str)

# Update CounterfactualDepthCalculator horizon
new_content = new_content.replace("horizon: int = 5) -> Tuple[int, float]:", "horizon: int = 10) -> Tuple[int, float]:")

# Update FreeWillIndex to use the new calculator
new_content = new_content.replace("self.veto_calc = VetoMechanism()", "self.veto_calc = VetoMechanism()\n        self.persistence_calc = TemporalPersistenceCalculator()")

# Update compute() method to include persistence
search_compute = "ma = self.compute_metacognitive_awareness(agent_state, prediction_error)"
replace_compute = search_compute + "\n\n        # 5. Temporal Persistence (P7)\n        persistence = self.persistence_calc.compute_persistence(agent_state, dynamics_model)"
new_content = new_content.replace(search_compute, replace_compute)

# Update FWI formula
search_fwi = "fwi = ("
replace_fwi = "fwi = (\n            self.weights.get('persistence', 0) * persistence +"
new_content = new_content.replace(search_fwi, replace_fwi)

# Update components dictionary
search_comps = "'external_constraint': float(ec)"
replace_comps = search_comps + ",\n                'persistence': float(persistence)"
new_content = new_content.replace(search_comps, replace_comps)

with open('free_will_framework.py', 'w') as f:
    f.write(new_content)
