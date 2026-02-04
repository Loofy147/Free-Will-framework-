import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

layers_code = """
class RealizationLayer:
    \"\"\"
    Defines a specific level of volitional actualization.
    \"\"\"
    INDIVIDUAL = 1
    BIOLOGICAL = 2
    SOCIAL     = 3
    TEMPORAL   = 4
    ETHICAL    = 5

class RealizationManager:
    \"\"\"
    Orchestrates the 'layers of realizations' in the free will framework.
    Each layer adds a new dimension of actualization to the agent.
    \"\"\"
    def __init__(self, fwi_calculator):
        self.fwi_calc = fwi_calculator
        self.active_layers = [
            RealizationLayer.INDIVIDUAL,
            RealizationLayer.BIOLOGICAL,
            RealizationLayer.SOCIAL,
            RealizationLayer.TEMPORAL,
            RealizationLayer.ETHICAL
        ]

    def realize_agency(self, agent_state, dynamics, conn, bounds, layer: int) -> Dict:
        \"\"\"
        Computes the FWI up to a specific realization layer.
        \"\"\"
        res = self.fwi_calc.compute(agent_state, dynamics, conn, bounds)

        # Layer-specific modulation
        if layer >= RealizationLayer.BIOLOGICAL:
            # Physical realization: Add metabolic cost / energy profile
            # High FWI requires more energy. Penalty = cost * FWI
            metabolic_cost = 0.02 * res['fwi']
            res['fwi'] = max(0.0, res['fwi'] - metabolic_cost)
            res['energy_fwi_ratio'] = res['fwi'] / (metabolic_cost + 1e-9)

        if layer < RealizationLayer.TEMPORAL:
            # Remove temporal components if not realized
            res['fwi'] -= (self.fwi_calc.weights.get('persistence', 0) * res['components'].get('persistence', 0))
            res['fwi'] += (self.fwi_calc.weights.get('volitional_integrity', 0) * (1.0 - res['components'].get('volitional_integrity', 1.0)))
            res['fwi'] = np.clip(res['fwi'], 0, 1)

        if layer < RealizationLayer.ETHICAL:
            # Remove moral alignment multiplier
            alignment = res['components'].get('moral_alignment', 1.0)
            if alignment > 0:
                res['fwi'] = res['fwi'] / alignment
            res['fwi'] = np.clip(res['fwi'], 0, 1)

        res['realization_layer'] = layer
        return res
"""

search_str = "class FreeWillIndex:"
new_content = content.replace(search_str, layers_code + "\n" + search_str)

with open('free_will_framework.py', 'w') as f:
    f.write(new_content)
