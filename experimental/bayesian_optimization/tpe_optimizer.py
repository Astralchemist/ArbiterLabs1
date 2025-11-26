import math
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class TPEOptimizer:
    """
    Tree-structured Parzen Estimator (TPE) Optimizer.
    """
    def __init__(self, search_space: Dict[str, Dict[str, Any]], optimize_mode: str = 'maximize', 
                 n_startup_jobs: int = 20, n_ei_candidates: int = 24, gamma: float = 0.25, seed: int = None):
        self.search_space = search_space
        self.optimize_mode = optimize_mode
        self.n_startup_jobs = n_startup_jobs
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)
        
        self.history: Dict[str, List[Dict[str, Any]]] = {k: [] for k in search_space}
        self.results: List[float] = []
        self.configs: List[Dict[str, Any]] = []

    def report_result(self, config: Dict[str, Any], result: float):
        if self.optimize_mode == 'maximize':
            loss = -result
        else:
            loss = result
            
        self.results.append(loss)
        self.configs.append(config)
        
        for key, value in config.items():
            self.history[key].append({'val': value, 'loss': loss})

    def suggest(self) -> Dict[str, Any]:
        config = {}
        for key, spec in self.search_space.items():
            config[key] = self._suggest_parameter(key, spec)
        return config

    def _suggest_parameter(self, key: str, spec: Dict[str, Any]) -> Any:
        param_history = self.history[key]
        
        if len(param_history) < self.n_startup_jobs:
            return self._random_sample(spec)
            
        if spec['type'] == 'choice':
            return self._suggest_categorical(spec, param_history)
        elif spec['type'] in ['uniform', 'loguniform', 'randint']:
            return self._suggest_numerical(spec, param_history)
        else:
            raise ValueError(f"Unknown parameter type: {spec['type']}")

    def _random_sample(self, spec: Dict[str, Any]) -> Any:
        if spec['type'] == 'uniform':
            return self.rng.uniform(spec['low'], spec['high'])
        elif spec['type'] == 'loguniform':
            return np.exp(self.rng.uniform(np.log(spec['low']), np.log(spec['high'])))
        elif spec['type'] == 'randint':
            return self.rng.randint(spec['low'], spec['high'])
        elif spec['type'] == 'choice':
            return self.rng.choice(spec['values'])
        else:
            raise ValueError(f"Unknown parameter type: {spec['type']}")

    def _suggest_categorical(self, spec: Dict[str, Any], history: List[Dict[str, Any]]) -> Any:
        values = spec['values']
        n_choices = len(values)
        
        losses = np.array([h['loss'] for h in history])
        vals = np.array([values.index(h['val']) for h in history])
        
        n_below = max(int(len(history) * self.gamma), 1)
        sorted_indices = np.argsort(losses)
        below_indices = sorted_indices[:n_below]
        above_indices = sorted_indices[n_below:]
        
        below_vals = vals[below_indices]
        above_vals = vals[above_indices]
        
        # Estimate probabilities
        # Simple count with smoothing
        prior_weight = 1.0
        
        below_counts = np.bincount(below_vals, minlength=n_choices) + prior_weight
        below_probs = below_counts / below_counts.sum()
        
        above_counts = np.bincount(above_vals, minlength=n_choices) + prior_weight
        above_probs = above_counts / above_counts.sum()
        
        # Expected Improvement ~ log(below_probs / above_probs)
        # We want to maximize this ratio
        score = np.log(below_probs) - np.log(above_probs)
        best_idx = np.argmax(score)
        
        return values[best_idx]

    def _suggest_numerical(self, spec: Dict[str, Any], history: List[Dict[str, Any]]) -> Any:
        losses = np.array([h['loss'] for h in history])
        vals = np.array([h['val'] for h in history])
        
        if spec['type'] == 'loguniform':
            vals = np.log(vals)
            low = np.log(spec['low'])
            high = np.log(spec['high'])
        else:
            low = spec['low']
            high = spec['high']
            
        n_below = max(int(len(history) * self.gamma), 1)
        sorted_indices = np.argsort(losses)
        below_indices = sorted_indices[:n_below]
        above_indices = sorted_indices[n_below:]
        
        below_vals = vals[below_indices]
        above_vals = vals[above_indices]
        
        # Adaptive Parzen Estimator
        # We construct a GMM for below and above
        
        mus_below = below_vals
        sigmas_below = self._calculate_sigmas(mus_below, low, high)
        
        mus_above = above_vals
        sigmas_above = self._calculate_sigmas(mus_above, low, high)
        
        # Sample candidates
        candidates = []
        for _ in range(self.n_ei_candidates):
            # Sample from below distribution (good samples)
            idx = self.rng.randint(len(mus_below))
            sample = self.rng.normal(mus_below[idx], sigmas_below[idx])
            # Clip
            sample = max(low, min(high, sample))
            candidates.append(sample)
            
        candidates = np.array(candidates)
        
        # Calculate log likelihoods
        lpdf_below = self._gmm_lpdf(candidates, mus_below, sigmas_below, low, high)
        lpdf_above = self._gmm_lpdf(candidates, mus_above, sigmas_above, low, high)
        
        # Maximize EI
        best_idx = np.argmax(lpdf_below - lpdf_above)
        best_val = candidates[best_idx]
        
        if spec['type'] == 'randint':
            return int(round(best_val))
        elif spec['type'] == 'loguniform':
            return float(np.exp(best_val))
        else:
            return float(best_val)

    def _calculate_sigmas(self, mus: np.ndarray, low: float, high: float) -> np.ndarray:
        # Simple rule of thumb for bandwidth
        # Sort mus
        order = np.argsort(mus)
        sorted_mus = mus[order]
        
        # Distances
        if len(mus) <= 1:
            return np.array([high - low]) * 0.2 # Default width
            
        densities = np.zeros_like(mus)
        
        # For inner points, use distance to neighbors
        densities[order[1:-1]] = np.maximum(
            sorted_mus[1:-1] - sorted_mus[:-2],
            sorted_mus[2:] - sorted_mus[1:-1]
        )
        
        # For boundary points
        densities[order[0]] = sorted_mus[1] - sorted_mus[0]
        densities[order[-1]] = sorted_mus[-1] - sorted_mus[-2]
        
        # Clip to reasonable range
        prior_sigma = (high - low)
        densities = np.clip(densities, prior_sigma * 0.01, prior_sigma)
        
        return densities

    def _gmm_lpdf(self, samples: np.ndarray, mus: np.ndarray, sigmas: np.ndarray, low: float, high: float) -> np.ndarray:
        # Calculate log pdf of samples under GMM defined by mus and sigmas
        # P(x) = (1/N) * sum(Normal(x; mu_i, sigma_i))
        
        # We need to handle the truncation at [low, high]
        # P_truncated(x) = P(x) / normalization
        # normalization = CDF(high) - CDF(low)
        
        # Vectorized calculation
        # samples: (M,)
        # mus: (N,)
        # sigmas: (N,)
        
        M = len(samples)
        N = len(mus)
        
        samples_exp = samples[:, np.newaxis] # (M, 1)
        mus_exp = mus[np.newaxis, :] # (1, N)
        sigmas_exp = sigmas[np.newaxis, :] # (1, N)
        
        # Normal PDF
        # (1 / (sigma * sqrt(2pi))) * exp(-0.5 * ((x - mu) / sigma)^2)
        norm_const = 1.0 / (sigmas_exp * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((samples_exp - mus_exp) / sigmas_exp) ** 2
        pdf = norm_const * np.exp(exponent) # (M, N)
        
        # Truncation normalization
        # CDF(x) = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
        def cdf(x, mu, sigma):
            return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))
            
        # Since math.erf doesn't support arrays, we iterate or use np.vectorize
        # Or we can use a loop since N is usually small (number of history points)
        
        # Let's use a simplified approach for normalization: assume it's 1 for simplicity if bounds are far
        # Or implement vectorized erf
        
        # Vectorized erf using numpy
        # np.vectorize(math.erf)
        vec_erf = np.vectorize(math.erf)
        
        cdf_high = 0.5 * (1 + vec_erf((high - mus_exp) / (sigmas_exp * np.sqrt(2))))
        cdf_low = 0.5 * (1 + vec_erf((low - mus_exp) / (sigmas_exp * np.sqrt(2))))
        normalization = cdf_high - cdf_low
        
        # Adjust pdf
        pdf = pdf / (normalization + 1e-10)
        
        # Average over mixture components
        avg_pdf = np.mean(pdf, axis=1)
        
        return np.log(avg_pdf + 1e-12)

