import numpy as np
import logging
from typing import List, Tuple, Optional, Union, Dict

logger = logging.getLogger(__name__)

class PortfolioOptimization:
    """
    Portfolio Optimization using Quantum-Inspired formulation (QUBO).
    
    Problem:
    Minimize: q * x^T * Sigma * x - mu^T * x
    Subject to: sum(x) = budget
    
    Converted to QUBO:
    Minimize: q * x^T * Sigma * x - mu^T * x + lambda * (sum(x) - budget)^2
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        covariances: np.ndarray,
        risk_factor: float,
        budget: int,
        penalty: float = 1000.0
    ) -> None:
        """
        Args:
            expected_returns: The expected returns for the assets.
            covariances: The covariances between the assets.
            risk_factor: The risk appetite of the decision maker (q).
            budget: The number of assets to be selected.
            penalty: Penalty coefficient for the budget constraint (lambda).
        """
        self.expected_returns = expected_returns
        self.covariances = covariances
        self.risk_factor = risk_factor
        self.budget = budget
        self.penalty = penalty
        self.num_assets = len(expected_returns)

    def get_qubo_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs the QUBO matrix Q and linear vector L.
        Energy = x^T Q x + L^T x + constant
        
        Returns:
            Q: Quadratic term matrix
            L: Linear term vector
        """
        # 1. Risk term: q * x^T * Sigma * x
        Q_risk = self.risk_factor * self.covariances
        
        # 2. Return term: -mu^T * x
        L_return = -self.expected_returns
        
        # 3. Constraint term: lambda * (sum(x) - B)^2
        # (sum(x) - B)^2 = (sum(x))^2 - 2B*sum(x) + B^2
        # (sum(x))^2 = sum_i sum_j x_i x_j
        # Since x_i is binary, x_i^2 = x_i. 
        # However, in standard QUBO x^T Q x, diagonal terms Q_ii handle x_i^2.
        
        # Matrix of ones for the quadratic part of the constraint
        J = np.ones((self.num_assets, self.num_assets))
        Q_constraint = self.penalty * J
        
        # Linear part of the constraint: -2 * lambda * B * sum(x)
        L_constraint = -2 * self.penalty * self.budget * np.ones(self.num_assets)
        
        # Combine terms
        Q = Q_risk + Q_constraint
        L = L_return + L_constraint
        
        # Note: In some QUBO formulations, linear terms are put on the diagonal of Q.
        # Here we keep them separate for clarity, but solvers might expect them merged.
        # Merging L into diagonal of Q:
        for i in range(self.num_assets):
            Q[i, i] += L[i]
            
        return Q

    def interpret_result(self, binary_solution: np.ndarray) -> Dict[str, Any]:
        """
        Interprets the binary solution.
        """
        selected_assets = [i for i, x in enumerate(binary_solution) if x == 1]
        
        portfolio_return = np.dot(self.expected_returns, binary_solution)
        portfolio_risk = np.dot(binary_solution, np.dot(self.covariances, binary_solution))
        
        return {
            "selected_assets": selected_assets,
            "expected_return": portfolio_return,
            "variance": portfolio_risk,
            "is_feasible": len(selected_assets) == self.budget
        }

class SimulatedAnnealingSolver:
    """
    A simple Simulated Annealing solver for QUBO problems.
    Minimize: x^T Q x
    """
    
    def __init__(self, n_iter: int = 1000, initial_temp: float = 10.0, cooling_rate: float = 0.99):
        self.n_iter = n_iter
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        
    def solve(self, Q: np.ndarray) -> np.ndarray:
        num_vars = Q.shape[0]
        
        # Random initialization
        current_state = np.random.randint(2, size=num_vars)
        current_energy = self._calculate_energy(Q, current_state)
        
        best_state = current_state.copy()
        best_energy = current_energy
        
        temp = self.initial_temp
        
        for _ in range(self.n_iter):
            # Propose a flip
            idx = np.random.randint(num_vars)
            new_state = current_state.copy()
            new_state[idx] = 1 - new_state[idx]
            
            # Calculate energy change efficiently
            # Delta E = E_new - E_old
            # This can be optimized, but full calculation is safer for now
            new_energy = self._calculate_energy(Q, new_state)
            
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / temp):
                current_state = new_state
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_state = current_state.copy()
                    best_energy = current_energy
            
            temp *= self.cooling_rate
            
        return best_state
    
    def _calculate_energy(self, Q: np.ndarray, x: np.ndarray) -> float:
        return np.dot(x, np.dot(Q, x))

def optimize_portfolio(expected_returns, covariances, risk_factor, budget):
    """
    Helper function to run the full optimization pipeline.
    """
    optimizer = PortfolioOptimization(expected_returns, covariances, risk_factor, budget)
    Q = optimizer.get_qubo_matrix()
    
    solver = SimulatedAnnealingSolver(n_iter=5000, initial_temp=100.0)
    solution = solver.solve(Q)
    
    result = optimizer.interpret_result(solution)
    return result
