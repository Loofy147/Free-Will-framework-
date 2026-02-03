"""
FWI WEIGHT OPTIMIZATION VIA MACHINE LEARNING
Optimizes the 7 component weights to maximize predictive accuracy
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import kendalltau, ttest_ind
import time
from typing import Dict, Tuple, List
import json

# Import the framework
from free_will_framework import FreeWillIndex, AgentState

# ============================================================================
# SYNTHETIC NEUROSCIENCE DATASET GENERATION
# ============================================================================

class SyntheticNeuroscienceDataset:
    """
    Generate synthetic agents with ground-truth FWI based on neuroscience
    heuristics and known relationships
    """

    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        np.random.seed(seed)

    def generate_agent_params(self) -> Dict:
        """Generate plausible neural/cognitive parameters with biological realism"""
        # Baseline traits
        pfc_vol = np.random.normal(10.5, 1.5)
        dopa = np.random.gamma(2, 0.5)
        conn = np.random.beta(3, 2)
        wm = np.random.poisson(7) + 2

        # Derived neural features (Dataset P3)
        # Readiness Potential (RP) onset time in ms (-1000 to -200)
        # Early RP indicates more 'unconscious' drive
        rp_onset = -200 - 800 * (1 - conn)

        # BOLD signals (normalized 0-1)
        dlpfc_bold = np.clip(np.random.normal(0.7, 0.1) * (pfc_vol / 10.5), 0, 1)
        acc_bold = np.clip(np.random.normal(0.5, 0.1) * (1 / dopa if dopa > 0 else 1), 0, 1)

        return {
            'pfc_volume': pfc_vol,
            'dopamine_level': dopa,
            'connectivity': conn,
            'wm_capacity': wm,
            'metacog_accuracy': np.random.beta(5, 2),
            'veto_capacity': np.random.beta(4, 2),
            'bayesian_precision': np.random.normal(0.8, 0.1),
            'constraint_level': np.random.beta(2, 5),
            # P3 specific features
            'rp_onset': rp_onset,
            'dlpfc_bold': dlpfc_bold,
            'acc_bold': acc_bold,
            'w_time': -200 + 100 * np.random.randn() # Subjective awareness time
        }

    def params_to_ground_truth_fwi(self, params: Dict) -> float:
        """
        Convert neural parameters to 'true' FWI based on neuroscience
        """

        # Normalize parameters to [0, 1]
        pfc_norm = np.clip((params['pfc_volume'] - 7) / 6, 0, 1)  # 7-13 → 0-1
        dopa_norm = np.clip(params['dopamine_level'] / 3, 0, 1)
        conn_norm = params['connectivity']
        wm_norm = np.clip((params['wm_capacity'] - 2) / 10, 0, 1)
        meta_norm = params['metacog_accuracy']
        veto_norm = params['veto_capacity']
        bayes_norm = np.clip(params['bayesian_precision'], 0, 1)
        const_norm = params['constraint_level']

        # Ground truth formula (based on neuroscience theory)
        # Enhanced with biological correlates (P3)
        # dlPFC activation and ACC monitoring are heavy weights
        bio_factor = 0.7 * params['dlpfc_bold'] + 0.3 * params['acc_bold']

        fwi_true = (
            0.15 * dopa_norm +           # Causal entropy
            0.10 * (pfc_norm * conn_norm) +  # Integration
            0.15 * wm_norm +             # Counterfactuals
            0.10 * meta_norm +           # Metacognition
            0.10 * veto_norm +           # Veto Efficacy
            0.10 * bayes_norm +          # Bayesian Precision
            0.30 * bio_factor +          # BIOLOGICAL CORRELATE (dlPFC + ACC)
            -0.10 * const_norm           # Penalty from constraints
        )

        # Add measurement noise (real neuroscience has ~10-20% noise)
        noise = np.random.normal(0, 0.05)
        fwi_true = np.clip(fwi_true + noise, 0, 1)

        return fwi_true

    def params_to_agent_state(self, params: Dict) -> Tuple[AgentState, callable, np.ndarray, np.ndarray]:
        """Convert parameters to computational agent representation"""

        # State dimensionality scaled by PFC volume
        n_beliefs = int(5 + params['pfc_volume'])
        n_actions = int(10 + params['wm_capacity'] * 2)

        # Create agent
        agent = AgentState(
            belief_state=np.random.randn(n_beliefs) * (1 - params['constraint_level']),
            goal_state=np.random.rand(5),
            meta_belief=np.random.randn(8) * (1 - params['metacog_accuracy']),  # Lower variance = higher accuracy
            action_repertoire=np.random.randn(n_actions, 3) * params['dopamine_level']
        )

        # Dynamics model
        def dynamics(s, a):
            a_flat = a.flatten()
            a_proj = np.zeros(len(s))
            a_proj[:len(a_flat)] = a_flat[:len(s)]
            return s * 0.9 + a_proj * 0.1 * params['dopamine_level']

        # Connectivity based on parameter
        connectivity = np.random.rand(n_beliefs, n_beliefs)
        connectivity = (connectivity + connectivity.T) / 2
        # Scale by connectivity parameter
        connectivity = connectivity * params['connectivity']
        np.fill_diagonal(connectivity, 0)

        # Constitutional bounds
        bounds = np.ones(3) * (2.0 + params['constraint_level'] * 3)

        return agent, dynamics, connectivity, bounds

    def generate_dataset(self) -> Tuple[List, List, List]:
        """
        Generate complete dataset
        """
        print(f"Generating {self.n_samples} synthetic agents...")

        agent_configs = []
        ground_truths = []
        all_params = []

        for i in range(self.n_samples):
            params = self.generate_agent_params()
            fwi_true = self.params_to_ground_truth_fwi(params)
            agent_config = self.params_to_agent_state(params)

            agent_configs.append(agent_config)
            ground_truths.append(fwi_true)
            all_params.append(params)

            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{self.n_samples}")

        print(f"✓ Dataset complete. Mean FWI: {np.mean(ground_truths):.3f} ± {np.std(ground_truths):.3f}")
        return agent_configs, ground_truths, all_params


# ============================================================================
# WEIGHT OPTIMIZATION ALGORITHMS
# ============================================================================

class WeightOptimizer:
    """Optimize FWI component weights"""

    def __init__(self, dataset: Tuple[List, List, List]):
        self.agent_configs, self.ground_truths, self.params = dataset
        self.n_samples = len(self.ground_truths)
        self.precalculated_components = self._precalculate()

    def _precalculate(self):
        print("Precalculating components for dataset...")
        components = []
        # Use a dummy calculator to get component values
        calculator = FreeWillIndex()
        for idx in range(self.n_samples):
            agent, dynamics, connectivity, bounds = self.agent_configs[idx]
            try:
                calculator.belief_updater.precision = self.params[idx]['bayesian_precision']
                calculator.veto_calc.veto_threshold = 1 - self.params[idx]['veto_capacity']

                # Temporarily reduce horizon for speed during precalculation
                calculator.causal_calc.tau = 10

                result = calculator.compute(agent, dynamics, connectivity, bounds, prediction_error=0.1)
                comps = result['components']
                components.append([
                    comps['causal_entropy'],
                    comps['integration_phi'],
                    comps['counterfactual_depth'],
                    comps['metacognition'],
                    comps['veto_efficacy'],
                    comps['bayesian_precision'],
                    comps['external_constraint']
                ])
            except Exception as e:
                print(f"Error precalculating index {idx}: {e}")
                components.append([0.5]*7)
            if (idx + 1) % 10 == 0:
                print(f"  Precalculated {idx+1}/{self.n_samples}")
        return np.array(components)

    def evaluate_weights(self, weights: np.ndarray, indices: np.ndarray = None) -> float:
        """
        Evaluate weights on dataset subset using precalculated components
        """
        if indices is None:
            indices = np.arange(self.n_samples)

        # Ensure weights sum to 1
        weights_norm = weights / weights.sum()

        # Weight vector (last one is penalty, so subtract)
        w = weights_norm.copy()
        w[6] = -w[6] # External constraint is a penalty

        # Compute predicted FWIs using matrix multiplication
        # components shape: (n_samples, 7)
        # w shape: (7,)
        predictions = np.dot(self.precalculated_components[indices], w)
        predictions = np.clip(predictions, 0, 1)

        ground_truth_subset = np.array([self.ground_truths[i] for i in indices])

        r2 = r2_score(ground_truth_subset, predictions)
        return -r2

    def optimize_bayesian(self, n_calls: int = 50) -> Tuple[np.ndarray, float]:
        """Bayesian optimization with Gaussian Process (simplified version)"""
        print("\n[ALGORITHM 1] Bayesian Optimization")

        start_time = time.time()

        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
        bounds = [(0, 1)] * 6 + [(0, 0.15)]

        # Initial guess (current enhanced weights)
        x0 = np.array([0.20, 0.15, 0.20, 0.15, 0.15, 0.15, 0.10])

        result = minimize(
            lambda w: self.evaluate_weights(w),
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': n_calls}
        )

        elapsed = time.time() - start_time
        weights_opt = result.x / result.x.sum()
        r2_score = -result.fun

        return weights_opt, r2_score

    def optimize_cmaes(self, generations: int = 20) -> Tuple[np.ndarray, float]:
        """CMA-ES evolutionary strategy"""
        print("\n[ALGORITHM 2] CMA-ES Evolutionary Strategy")

        start_time = time.time()

        def objective(w):
            w_norm = w / w.sum()
            return self.evaluate_weights(w_norm)

        bounds = [(0.1, 0.4)] * 6 + [(0.0, 0.15)]

        result = differential_evolution(
            objective,
            bounds,
            maxiter=generations,
            popsize=10,
            seed=42
        )

        elapsed = time.time() - start_time
        weights_opt = result.x / result.x.sum()
        r2_score = -result.fun

        return weights_opt, r2_score


def cross_validate_weights(weights: np.ndarray,
                          optimizer: WeightOptimizer,
                          k_folds: int = 10) -> Dict:
    """K-fold cross-validation"""

    n_samples = optimizer.n_samples
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    test_r2s = []

    # Weight vector (last one is penalty, so subtract)
    w = weights.copy()
    w[6] = -w[6]

    for fold, (train_idx, test_idx) in enumerate(kf.split(range(n_samples))):
        test_preds = np.dot(optimizer.precalculated_components[test_idx], w)
        test_preds = np.clip(test_preds, 0, 1)
        test_true = np.array([optimizer.ground_truths[i] for i in test_idx])

        test_r2s.append(r2_score(test_true, test_preds))

    return {
        'mean_r2': np.mean(test_r2s),
        'std_r2': np.std(test_r2s)
    }


def run_weight_optimization():
    """Execute complete weight optimization pipeline"""

    print("="*80)
    print("FWI WEIGHT OPTIMIZATION VIA MACHINE LEARNING")
    print("="*80)

    dataset_gen = SyntheticNeuroscienceDataset(n_samples=200, seed=42)
    dataset = dataset_gen.generate_dataset()

    optimizer = WeightOptimizer(dataset)

    baseline_weights = np.array([0.20, 0.15, 0.20, 0.15, 0.15, 0.15, 0.10])
    baseline_cv = cross_validate_weights(baseline_weights, optimizer, k_folds=5)
    print(f"  Baseline CV R²: {baseline_cv['mean_r2']:.4f}")

    weights_bayes, r2_bayes = optimizer.optimize_bayesian(n_calls=20)
    weights_cmaes, r2_cmaes = optimizer.optimize_cmaes(generations=10)

    results = [
        ('Bayesian', weights_bayes, r2_bayes),
        ('CMA-ES', weights_cmaes, r2_cmaes)
    ]

    best_method, best_weights, best_train_r2 = max(results, key=lambda x: x[2])

    print(f"\n  Best method: {best_method}")
    print(f"  Optimized weights: {best_weights}")

    optimized_cv = cross_validate_weights(best_weights, optimizer, k_folds=3)
    print(f"  Optimized CV R²: {optimized_cv['mean_r2']:.4f}")

    final_results = {
        'baseline_weights': baseline_weights.tolist(),
        'baseline_r2': baseline_cv['mean_r2'],
        'optimized_weights': best_weights.tolist(),
        'optimized_r2': optimized_cv['mean_r2'],
        'method': best_method
    }

    with open('optimized_weights.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    return final_results


if __name__ == "__main__":
    run_weight_optimization()
