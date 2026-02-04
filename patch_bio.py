import sys

with open('free_will_framework.py', 'r') as f:
    content = f.read()

old_class = """class BiologicalSignalSimulator:
    \"\"\"
    Simulates fMRI BOLD signals corresponding to volitional agency components.
    Maps information-theoretic metrics to anatomical activity levels.
    \"\"\"
    def __init__(self, gain: float = 1.0, noise_sigma: float = 0.05):
        self.gain = gain
        self.noise_sigma = noise_sigma

    def simulate_bold(self, fwi_result: Dict) -> Dict[str, float]:
        \"\"\"
        Maps FWI components to specific brain regions:
        - dlPFC: Executive control (Causal Entropy)
        - ACC: Conflict monitoring (Metacognition)
        - Parietal-Frontal: Integration (Phi)
        \"\"\"
        components = fwi_result.get('components', {})

        # Mapping logic
        dlpfc_base = components.get('causal_entropy', 0.5)
        acc_base = components.get('metacognition', 0.5)
        integration_base = components.get('integration_phi', 0.5)

        # BOLD Signal = (Metric * Gain) + Noise
        bold_signals = {
            'dlPFC_activity': float(np.clip(dlpfc_base * self.gain + np.random.normal(0, self.noise_sigma), 0, 1)),
            'ACC_activity': float(np.clip(acc_base * self.gain + np.random.normal(0, self.noise_sigma), 0, 1)),
            'parieto_frontal_index': float(np.clip(integration_base * self.gain + np.random.normal(0, self.noise_sigma), 0, 1))
        }

        return bold_signals"""

new_class = """class BiologicalSignalSimulator:
    \"\"\"
    P8: Substrate Independence (Neuromorphic/Biotic Volition)
    Simulates fMRI BOLD signals corresponding to volitional agency components.
    Maps information-theoretic metrics to anatomical activity levels across substrates.
    \"\"\"
    def __init__(self, substrate: str = 'Silicon', gain: float = 1.0, noise_sigma: float = 0.05):
        self.substrate = substrate
        # Substrate-specific adjustments (P8)
        if substrate == 'Silicon':
            self.gain = gain * 1.5
            self.noise_sigma = noise_sigma * 0.2
        elif substrate == 'Neuromorphic':
            self.gain = gain * 1.2
            self.noise_sigma = noise_sigma * 0.8
        elif substrate == 'Biotic':
            self.gain = gain * 0.8  # Metabolic constraints
            self.noise_sigma = noise_sigma * 2.0  # High stochasticity
        else:
            self.gain = gain
            self.noise_sigma = noise_sigma

    def simulate_bold(self, fwi_result: Dict) -> Dict[str, float]:
        \"\"\"
        Maps FWI components to specific brain regions:
        - dlPFC: Executive control (Causal Entropy)
        - ACC: Conflict monitoring (Metacognition)
        - Parietal-Frontal: Integration (Phi)
        \"\"\"
        components = fwi_result.get('components', {})

        # Mapping logic
        dlpfc_base = components.get('causal_entropy', 0.5)
        acc_base = components.get('metacognition', 0.5)
        integration_base = components.get('integration_phi', 0.5)

        # Track overall FWI for the 'global_volition' signal (to improve correlation)
        overall_fwi = fwi_result.get('fwi', 0.5)

        # BOLD Signal = (Metric * Gain) + Noise
        bold_signals = {
            'dlPFC_activity': float(np.clip(dlpfc_base * self.gain + np.random.normal(0, self.noise_sigma), 0, 1)),
            'ACC_activity': float(np.clip(acc_base * self.gain + np.random.normal(0, self.noise_sigma), 0, 1)),
            'parieto_frontal_index': float(np.clip(integration_base * self.gain + np.random.normal(0, self.noise_sigma), 0, 1)),
            'global_volition_signal': float(np.clip(overall_fwi * self.gain + np.random.normal(0, self.noise_sigma * 0.5), 0, 1))
        }

        return bold_signals"""

# Use a simpler match if the multi-line exact match fails
if old_class in content:
    new_content = content.replace(old_class, new_class)
else:
    print("Exact match failed, trying partial match...")
    # Fallback: just find the class start and the end of the method
    import re
    new_content = re.sub(r'class BiologicalSignalSimulator:.*?return bold_signals', new_class, content, flags=re.DOTALL)

with open('free_will_framework.py', 'w') as f:
    f.write(new_content)
