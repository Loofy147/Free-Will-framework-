import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

energy_code = """    def compute_energy_cost(self, fwi_result: Dict) -> Dict[str, float]:
        \"\"\"
        P8: Cost of Choice
        Quantifies bits of freedom per Joule (simulated).
        Silicon: Efficient but rigid.
        Biotic: High cost, high stochasticity.
        \"\"\"
        fwi = fwi_result.get('fwi', 0.5)

        # Simulated Energy in Joules
        if self.substrate == 'Silicon':
            joules = 0.001 * fwi
        elif self.substrate == 'Neuromorphic':
            joules = 0.0001 * fwi  # 10x more efficient
        elif self.substrate == 'Biotic':
            joules = 0.01 * fwi    # Expensive
        else:
            joules = 0.005 * fwi

        return {
            'joules_per_choice': float(joules),
            'energy_fwi_ratio': float(fwi / (joules + 1e-9))
        }
"""

search_str = "return bold_signals"
content = content.replace(search_str, search_str + "\n\n" + energy_code)

with open('free_will_framework.py', 'w') as f:
    f.write(content)
