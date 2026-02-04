import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

landauer_code = """    def compute_energy_cost(self, fwi_result: Dict) -> Dict[str, float]:
        \"\"\"
        P8: Volitional Thermodynamics (Landauer's Principle)
        E >= k_B * T * ln(2) per bit of information erased/processed in choice.
        \"\"\"
        fwi = fwi_result.get('fwi', 0.5)
        ce = fwi_result['components'].get('causal_entropy', 0.5)

        # Physical Constants
        k_B = 1.380649e-23  # Boltzmann constant
        T = 310.15          # Biological temperature (37°C)
        ln2 = 0.693147

        landauer_limit = k_B * T * ln2

        # Estimated bits of freedom based on causal entropy
        # ce is log(N_reachable), so bits = ce / ln(2)
        volitional_bits = ce / ln2

        min_energy_joules = landauer_limit * volitional_bits

        # Substrate efficiency factors
        if self.substrate == 'Silicon':
            efficiency = 1e-9  # Modern CPUs are ~10^9 times landauer limit
        elif self.substrate == 'Neuromorphic':
            efficiency = 1e-6  # 1000x more efficient than silicon
        elif self.substrate == 'Biotic':
            efficiency = 1e-3  # Metabolic overhead
        else:
            efficiency = 1e-7

        actual_energy = min_energy_joules / (efficiency + 1e-15)

        return {
            'landauer_limit_joules': float(min_energy_joules),
            'actual_energy_joules': float(actual_energy),
            'volitional_bits': float(volitional_bits),
            'energy_fwi_ratio': float(fwi / (actual_energy + 1e-18))
        }
"""

import re
# Replace the old compute_energy_cost method
content = re.sub(r'def compute_energy_cost\(self, fwi_result: Dict\) -> Dict\[str, float\]:.*?return \{.*?\}', landauer_code, content, flags=re.DOTALL)

with open('free_will_framework.py', 'w') as f:
    f.write(content)
