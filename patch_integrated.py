import sys

with open('integrated_framework.py', 'r') as f:
    content = f.read()

# Update weights in __init__
weights_search = """        self.fwi_calc.weights = {
            'causal_entropy': 0.10,
            'integration': 0.30,
            'counterfactual': 0.40,
            'metacognition': 0.05,
            'veto_efficacy': 0.05,
            'bayesian_precision': 0.05,
            'persistence': 0.10,
            'constraint_penalty': 0.0
        }"""

weights_replace = """        self.fwi_calc.weights = {
            'causal_entropy': 0.10,
            'integration': 0.30,
            'counterfactual': 0.40,
            'metacognition': 0.05,
            'veto_efficacy': 0.05,
            'bayesian_precision': 0.05,
            'persistence': 0.05,
            'volitional_integrity': 0.05,
            'constraint_penalty': 0.0
        }"""
content = content.replace(weights_search, weights_replace)

# Update bio_sim init
bio_search = "self.bio_sim = BiologicalSignalSimulator(gain=1.2, noise_sigma=0.03)"
bio_replace = "self.bio_sim = BiologicalSignalSimulator(substrate='Neuromorphic', gain=1.2, noise_sigma=0.03)"
content = content.replace(bio_search, bio_replace)

# Update global_benchmark to 50 steps
benchmark_search = "def global_benchmark(self, n_agents: int = 10):"
benchmark_replace = "def global_benchmark(self, n_agents: int = 10, n_steps: int = 50):"
content = content.replace(benchmark_search, benchmark_replace)

# Update benchmark implementation to use steps and calculate correlation matrix
# I'll just replace the whole method for simplicity
import re
method_pattern = r'def global_benchmark\(self, n_agents: int = 10, n_steps: int = 50\):.*?return report'
new_method = """def global_benchmark(self, n_agents: int = 10, n_steps: int = 50):
        \"\"\"Runs the Global Volition Benchmark over multiple steps\"\"\"
        print("\n" + "="*70)
        print(f" GLOBAL VOLITION BENCHMARK ({n_agents} agents, {n_steps} steps)")
        print("="*70)

        agents = []
        for i in range(n_agents):
            agent = AgentState(
                belief_state=np.random.randn(10),
                goal_state=np.random.rand(5),
                meta_belief=np.random.randn(8) * 0.5,
                action_repertoire=np.random.randn(20, 3)
            )
            agents.append(agent)

        def dynamics(s, a):
            a_flat = a.flatten()
            a_proj = np.zeros(len(s))
            a_proj[:len(a_flat)] = a_flat[:len(s)]
            return 0.9 * s + 0.1 * a_proj
        bounds = np.ones(3) * 2.0

        step_data = []

        for t in range(n_steps):
            individual_fwis = []
            bold_signals = []

            for agent in agents:
                res = self.compute_full_agency(agent, dynamics, np.eye(10), bounds)
                individual_fwis.append(res['fwi'])
                bold_signals.append(res['biological_signals']['global_volition_signal'])
                # Evolve agent belief slightly for next step
                agent.belief_state = dynamics(agent.belief_state, agent.action_repertoire[0])

            # Social coupling
            coupling = np.random.rand(n_agents, n_agents)
            coupling = (coupling + coupling.T) / 2
            social_res = self.simulate_collective_volition(agents, coupling)

            step_report = {
                't': t,
                'mean_fwi': np.mean(individual_fwis),
                'collective_fwi': social_res['collective_fwi'],
                'dv': social_res['democratic_volition'],
                'bold_corr': np.corrcoef(individual_fwis, bold_signals)[0, 1] if len(individual_fwis) > 1 else 1.0
            }
            step_data.append(step_report)
            if t % 10 == 0:
                print(f"   Step {t:2d}: Mean FWI={step_report['mean_fwi']:.4f}, BOLD Corr={step_report['bold_corr']:.4f}")

        # Final Correlation Matrix: Individual FWI vs Social Synergy vs BOLD Fidelity
        # We'll use the means over time for the matrix
        final_fwis = [d['mean_fwi'] for d in step_data]
        final_social = [d['collective_fwi'] for d in step_data]
        final_bold = [d['bold_corr'] for d in step_data]

        corr_matrix = np.corrcoef([final_fwis, final_social, final_bold]).tolist()

        report = {
            'mean_individual_fwi': float(np.mean(final_fwis)),
            'collective_fwi': float(np.mean(final_social)),
            'democratic_volition': float(np.mean([d['dv'] for d in step_data])),
            'fwi_bold_correlation': float(np.mean(final_bold)),
            'volition_correlation_matrix': corr_matrix,
            'steps_completed': n_steps
        }

        self.status_report['global_benchmark'] = report
        print(f"\n   BENCHMARK COMPLETE")
        print(f"   Average BOLD Correlation: {report['fwi_bold_correlation']:.4f}")
        return report"""

content = re.sub(method_pattern, new_method, content, flags=re.DOTALL)

with open('integrated_framework.py', 'w') as f:
    f.write(content)
