from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


class PersistenceModel:
    """Naive baseline: forecast equals lag_1."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PersistenceModel":
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x[:, 0]


class LinearBaseline:
    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearBaseline":
        self.model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


class MLPBaseline:
    def __init__(self) -> None:
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=400,
            random_state=42,
            learning_rate_init=1e-3,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPBaseline":
        self.model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


@dataclass
class NeuroFuzzyConfig:
    n_rules: int = 4
    random_state: int = 42


class NeuroFuzzyRegressor:
    """Simple first-order TS neuro-fuzzy model with Gaussian memberships.

    We use two core inputs for fuzzy premises (lag_1 and lag_24) and then
    combine rule activations with linear consequents.
    """

    def __init__(self, config: NeuroFuzzyConfig | None = None):
        self.config = config or NeuroFuzzyConfig()
        self.params_: np.ndarray | None = None
        self.w_: np.ndarray | None = None

    def _unpack(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.config.n_rules
        centers = p[: 2 * r].reshape(r, 2)
        sigmas = np.exp(p[2 * r : 4 * r]).reshape(r, 2) + 1e-3
        return centers, sigmas

    def _firing(self, x2: np.ndarray, p: np.ndarray) -> np.ndarray:
        centers, sigmas = self._unpack(p)
        out = []
        for j in range(self.config.n_rules):
            z = ((x2 - centers[j]) / sigmas[j]) ** 2
            out.append(np.exp(-0.5 * np.sum(z, axis=1)))
        f = np.vstack(out).T
        f = f / (f.sum(axis=1, keepdims=True) + 1e-9)
        return f

    def _design(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        x2 = x[:, [0, 3]]  # lag_1, lag_24
        fire = self._firing(x2, p)
        # Rule-wise linear consequents over all features + bias
        n, d = x.shape
        mats = []
        xb = np.hstack([x, np.ones((n, 1))])
        for j in range(self.config.n_rules):
            mats.append(fire[:, [j]] * xb)
        return np.hstack(mats)

    def _fit_consequents(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        phi = self._design(x, p)
        reg = 1e-3
        a = phi.T @ phi + reg * np.eye(phi.shape[1])
        b = phi.T @ y
        return np.linalg.solve(a, b)

    def _loss(self, p: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        w = self._fit_consequents(x, y, p)
        pred = self._design(x, p) @ w
        return float(np.mean((pred - y) ** 2))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NeuroFuzzyRegressor":
        rng = np.random.default_rng(self.config.random_state)
        r = self.config.n_rules
        x2 = x[:, [0, 3]]
        idx = rng.choice(len(x2), size=r, replace=False)
        centers0 = x2[idx].reshape(-1)
        sig0 = np.log(np.full(2 * r, x2.std() + 1.0))
        p0 = np.concatenate([centers0, sig0])
        res = minimize(self._loss, p0, args=(x, y), method="L-BFGS-B", options={"maxiter": 120})
        self.params_ = res.x
        self.w_ = self._fit_consequents(x, y, self.params_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.params_ is None or self.w_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return self._design(x, self.params_) @ self.w_


class PSOOptimizedNeuroFuzzy(NeuroFuzzyRegressor):
    """PSO-optimized premise parameters; consequents solved by ridge LS."""

    def __init__(self, config: NeuroFuzzyConfig | None = None, n_particles: int = 20, iters: int = 40):
        super().__init__(config=config)
        self.n_particles = n_particles
        self.iters = iters

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PSOOptimizedNeuroFuzzy":
        # Baseline ANFIS-like fit as fallback for stability.
        baseline = NeuroFuzzyRegressor(config=self.config).fit(x, y)
        base_params = baseline.params_.copy()
        base_w = baseline.w_.copy()
        base_loss = np.mean((baseline.predict(x) - y) ** 2)

        rng = np.random.default_rng(self.config.random_state)
        r = self.config.n_rules
        dim = 4 * r
        x2 = x[:, [0, 3]]
        low = np.array([x2[:, 0].min(), x2[:, 1].min()] * r + [math.log(1.0)] * (2 * r))
        high = np.array([x2[:, 0].max(), x2[:, 1].max()] * r + [math.log(x2.std() + 5.0)] * (2 * r))

        pos = rng.uniform(low, high, size=(self.n_particles, dim))
        vel = rng.normal(0, 0.2, size=(self.n_particles, dim))
        pbest = pos.copy()
        pbest_score = np.array([self._loss(p, x, y) for p in pos])
        g_idx = int(np.argmin(pbest_score))
        gbest = pbest[g_idx].copy()

        w_inertia, c1, c2 = 0.72, 1.35, 1.35
        for _ in range(self.iters):
            r1 = rng.random((self.n_particles, dim))
            r2 = rng.random((self.n_particles, dim))
            vel = (
                w_inertia * vel
                + c1 * r1 * (pbest - pos)
                + c2 * r2 * (gbest[None, :] - pos)
            )
            pos = np.clip(pos + vel, low, high)
            score = np.array([self._loss(p, x, y) for p in pos])
            improve = score < pbest_score
            pbest[improve] = pos[improve]
            pbest_score[improve] = score[improve]
            gbest = pbest[int(np.argmin(pbest_score))].copy()

        self.params_ = gbest
        self.w_ = self._fit_consequents(x, y, self.params_)
        pso_loss = np.mean((self.predict(x) - y) ** 2)
        if pso_loss > base_loss:
            self.params_ = base_params
            self.w_ = base_w
        return self
