import numpy as np
from copy import deepcopy

class Categorical(object):
    """
    Categorical distribution for categorical variables parametrized by theta.
    """

    def __init__(self, categories):
        self.d = len(categories)
        self.C = categories
        self.Cmax = np.max(categories)
        self.theta = np.zeros((self.d, self.Cmax))
        # initialize theta by 1/C for each dimensions
        for i in range(self.d):
            self.theta[i, :self.C[i]] = 1./self.C[i]
        # pad zeros to unused elements
        for i in range(self.d):
            self.theta[i, self.C[i]:] = 0.

    def sampling(self):
        """
        Draw a sample from the categorical distribution.
        :return: sampled variables from the categorical distribution (one-hot representation)
        """
        rand = np.random.rand(self.d, 1)    # range of random number is [0, 1)
        cum_theta = self.theta.cumsum(axis=1)    # (d, Cmax)

        # x[i, j] becomes 1 iff cum_theta[i, j] - theta[i, j] <= rand[i] < cum_theta[i, j]
        x = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        return x

    def mle(self):
        """
        Return the most likely categories.
        :return: categorical variables (one-hot representation)
        """
        m = self.theta.argmax(axis=1)
        x = np.zeros((self.d, self.Cmax))
        for i, c in enumerate(m):
            x[i, c] = 1
        return x

def one_hot_to_index(one_hot_matrix):
    return np.array([np.where(r == 1)[0][0] for r in one_hot_matrix])

def index_to_one_hot(index_vector, C):
    return np.eye(C)[index_vector.reshape(-1)]

class StochasticNaturalGradientOptimizer:
    """
    Adaptive Stochastic Natural Gradient (ASNG) for Categorical Distribution.
    Adapted from XNAS.
    """

    def __init__(self, categories,
                 alpha=1.5, delta_init=1., lam=6,
                 Delta_max=np.inf, init_theta=None, maximize=True):
        """
        categories: list of integers, number of categories for each dimension.
        alpha: threshold for adaptation
        delta_init: initial learning rate
        lam: lambda, sample size for update
        Delta_max: maximum Delta
        maximize: True for maximization, False for minimization
        """

        self.N = np.sum(np.array(categories) - 1)
        # Categorical distribution
        self.p_model = Categorical(categories)
        # valid dimension size
        self.p_model.C = np.array(self.p_model.C)
        self.valid_d = len(self.p_model.C[self.p_model.C > 1])

        if init_theta is not None:
            self.p_model.theta = init_theta

        # Adaptive SG
        self.alpha = alpha  # threshold for adaptation
        self.delta_init = delta_init
        self.lam = lam  # lambda_theta
        self.Delta_max = Delta_max  # maximum Delta (can be np.inf)

        self.Delta = 1.
        self.gamma = 0.0  # correction factor
        self.s = np.zeros(self.N)  # averaged stochastic natural gradient
        self.delta = self.delta_init / self.Delta
        self.eps = self.delta

        self.sample = []
        self.objective = []
        self.maximize = 1 if maximize else -1

    def record_information(self, sample, objective):
        """
        Record a sample and its objective value.
        sample: one-hot encoded sample
        objective: float value
        """
        self.sample.append(sample)
        self.objective.append(objective * self.maximize)

    def sampling(self):
        """
        Draw a sample from the categorical distribution (one-hot)
        """
        rand = np.random.rand(self.p_model.d, 1)  # range of random number is [0, 1)
        cum_theta = self.p_model.theta.cumsum(axis=1)  # (d, Cmax)

        # x[i, j] becomes 1 if cum_theta[i, j] - theta[i, j] <= rand[i] < cum_theta[i, j]
        c = (cum_theta - self.p_model.theta <= rand) & (rand < cum_theta)
        return c

    def sampling_index(self):
        return one_hot_to_index(np.array(self.sampling()))

    def mle(self):
        """
        Get most likely categorical variables (one-hot)
        """
        m = self.p_model.theta.argmax(axis=1)
        x = np.zeros((self.p_model.d, self.p_model.Cmax))
        for i, c in enumerate(m):
            x[i, c] = 1
        return x

    def step(self):
        """
        Perform an update step if enough samples have been collected.
        """
        if len(self.sample) >= self.lam:
            objective = np.array(self.objective)
            sample_array = np.array(self.sample)
            self._update_function(sample_array, objective)
            self.sample = []
            self.objective = []

    def _update_function(self, c_one, fxc, range_restriction=True):
        self.delta = self.delta_init / self.Delta
        beta = self.delta / (self.N ** 0.5)

        aru, idx = self._utility(fxc)
        if np.all(aru == 0):
            return

        ng = np.mean(aru[:, np.newaxis, np.newaxis] * (c_one[idx] - self.p_model.theta), axis=0)

        sl = []
        for i, K in enumerate(self.p_model.C):
            theta_i = self.p_model.theta[i, :K - 1]
            theta_K = self.p_model.theta[i, K - 1]
            s_i = 1. / np.sqrt(theta_i) * ng[i, :K - 1]
            s_i += np.sqrt(theta_i) * ng[i, :K - 1].sum() / (theta_K + np.sqrt(theta_K))
            sl += list(s_i)
        sl = np.array(sl)

        pnorm = np.sqrt(np.dot(sl, sl)) + 1e-9
        self.eps = self.delta / pnorm
        self.p_model.theta += self.eps * ng

        self.s = (1 - beta) * self.s + np.sqrt(beta * (2 - beta)) * sl / pnorm
        self.gamma = (1 - beta) ** 2 * self.gamma + beta * (2 - beta)
        self.Delta *= np.exp(beta * (self.gamma - np.dot(self.s, self.s) / self.alpha))
        self.Delta = min(self.Delta, self.Delta_max)

        for i in range(self.p_model.d):
            ci = self.p_model.C[i]
            # Constraint for theta (minimum value of theta and sum of theta = 1.0)
            theta_min = 1. / (self.valid_d * (ci - 1)) if range_restriction and ci > 1 else 0.
            self.p_model.theta[i, :ci] = np.maximum(self.p_model.theta[i, :ci], theta_min)
            theta_sum = self.p_model.theta[i, :ci].sum()
            tmp = theta_sum - theta_min * ci
            self.p_model.theta[i, :ci] -= (theta_sum - 1.) * (self.p_model.theta[i, :ci] - theta_min) / tmp
            # Ensure the summation to 1
            self.p_model.theta[i, :ci] /= self.p_model.theta[i, :ci].sum()

    @staticmethod
    def _utility(f, rho=0.25, negative=True):
        """
        Ranking Based Utility Transformation
        """
        eps = 1e-14
        idx = np.argsort(f)
        lam = len(f)
        mu = int(np.ceil(lam * rho))
        _w = np.zeros(lam)
        _w[:mu] = 1 / mu
        _w[lam - mu:] = -1 / mu if negative else 0
        w = np.zeros(lam)
        istart = 0
        for i in range(f.shape[0] - 1):
            if f[idx[i + 1]] - f[idx[i]] < eps * f[idx[i]]:
                pass
            elif istart < i:
                w[istart:i + 1] = np.mean(_w[istart:i + 1])
                istart = i + 1
            else:
                w[i] = _w[i]
                istart = i + 1
        w[istart:] = np.mean(_w[istart:])
        return w, idx
