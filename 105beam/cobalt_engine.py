"""Paper-aligned COBALT optimizer.

The implementation uses a uniformly sampled spanning tree, an additive
Matern-5/2 kernel over tree edges, a fully Bayesian SAAS prior fitted by NUTS,
and graph-only trust-region acquisition search.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pyro
import torch
from pyro.infer import MCMC, NUTS


Tensor = torch.Tensor


def uniform_spanning_tree(n_variables: int, seed: int) -> tuple[tuple[int, int], ...]:
    """Sample a uniform labelled tree through a uniform Prüfer sequence."""
    if n_variables < 2:
        raise ValueError("a spanning tree requires at least two variables")
    rng = np.random.default_rng(seed)
    prufer = rng.integers(0, n_variables, size=n_variables - 2)
    degree = np.ones(n_variables, dtype=int)
    for node in prufer:
        degree[int(node)] += 1
    leaves = [int(i) for i in np.flatnonzero(degree == 1)]
    heapq.heapify(leaves)
    edges: list[tuple[int, int]] = []
    for node in prufer:
        leaf = heapq.heappop(leaves)
        node = int(node)
        edges.append((min(leaf, node), max(leaf, node)))
        degree[leaf] -= 1
        degree[node] -= 1
        if degree[node] == 1:
            heapq.heappush(leaves, node)
    u = heapq.heappop(leaves)
    v = heapq.heappop(leaves)
    edges.append((min(u, v), max(u, v)))
    result = tuple(edges)
    if len(result) != n_variables - 1:
        raise RuntimeError("invalid Prüfer tree")
    return result


def validate_tree(edges: Iterable[tuple[int, int]], n_variables: int) -> None:
    edges = tuple(edges)
    if len(edges) != n_variables - 1:
        raise ValueError("tree must contain e-1 edges")
    parent = list(range(n_variables))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        if not (0 <= u < n_variables and 0 <= v < n_variables and u != v):
            raise ValueError("invalid tree edge")
        ru, rv = find(u), find(v)
        if ru == rv:
            raise ValueError("cycle detected")
        parent[ru] = rv
    if len({find(i) for i in range(n_variables)}) != 1:
        raise ValueError("tree is disconnected")


def _edge_dims(edge: tuple[int, int], latent_dim: int) -> tuple[int, ...]:
    u, v = edge
    return tuple(range(u * latent_dim, (u + 1) * latent_dim)) + tuple(
        range(v * latent_dim, (v + 1) * latent_dim)
    )


def _matern52(x1: Tensor, x2: Tensor, lengthscale: Tensor) -> Tensor:
    delta = (x1[:, None, :] - x2[None, :, :]) / lengthscale
    radius = torch.sqrt(torch.clamp((delta * delta).sum(dim=-1), min=1.0e-18))
    root5 = math.sqrt(5.0) * radius
    return (1.0 + root5 + 5.0 * radius.square() / 3.0) * torch.exp(-root5)


def _tree_kernel(
    x1: Tensor,
    x2: Tensor,
    edges: tuple[tuple[int, int], ...],
    latent_dim: int,
    lengthscale: Tensor,
    outputscale: Tensor,
) -> Tensor:
    return outputscale * _tree_kernel_components(
        x1, x2, edges, latent_dim, lengthscale
    ).sum(dim=0)


def _tree_kernel_components(
    x1: Tensor,
    x2: Tensor,
    edges: tuple[tuple[int, int], ...],
    latent_dim: int,
    lengthscale: Tensor,
) -> Tensor:
    """Evaluate all edge kernels as one E-by-N1-by-N2 tensor."""
    dimensions = [_edge_dims(edge, latent_dim) for edge in edges]
    first = torch.stack([x1[:, dims] for dims in dimensions], dim=0)
    second = torch.stack([x2[:, dims] for dims in dimensions], dim=0)
    scales = torch.stack([lengthscale[list(dims)] for dims in dimensions], dim=0)
    delta = (first[:, :, None, :] - second[:, None, :, :]) / scales[:, None, None, :]
    radius = torch.sqrt(torch.clamp(delta.square().sum(dim=-1), min=1.0e-18))
    root5 = math.sqrt(5.0) * radius
    return (1.0 + root5 + 5.0 * radius.square() / 3.0) * torch.exp(-root5)


@dataclass(frozen=True)
class NutsConfig:
    warmup_steps: int = 64
    num_samples: int = 32
    thinning: int = 2
    max_tree_depth: int = 4


class AdditiveTreeSaasGP:
    """Fully Bayesian additive SAAS-GP for one sampled tree."""

    def __init__(
        self,
        n_variables: int,
        latent_dim: int,
        edges: tuple[tuple[int, int], ...],
        nuts: NutsConfig,
        device: torch.device,
        seed: int,
    ) -> None:
        validate_tree(edges, n_variables)
        self.n_variables = n_variables
        self.latent_dim = latent_dim
        self.edges = edges
        self.nuts = nuts
        self.device = device
        self.seed = seed
        self.samples: dict[str, Tensor] | None = None
        self.x_train: Tensor | None = None
        self.y_train: Tensor | None = None
        self.yvar_train: Tensor | None = None
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.fit_seconds = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray, yvar: np.ndarray) -> None:
        x_np = np.asarray(x, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64).reshape(-1)
        yvar_np = np.asarray(yvar, dtype=np.float64).reshape(-1)
        if x_np.shape != (len(y_np), self.n_variables * self.latent_dim):
            raise ValueError("training input shape does not match the tree domain")
        if len(yvar_np) != len(y_np) or np.any(yvar_np <= 0.0):
            raise ValueError("heteroscedastic variances must be positive")
        self.y_mean = float(y_np.mean())
        self.y_scale = float(y_np.std(ddof=1)) if len(y_np) > 1 else 1.0
        self.y_scale = max(self.y_scale, 1.0e-8)
        self.x_train = torch.as_tensor(x_np, dtype=torch.double, device=self.device)
        self.y_train = torch.as_tensor(
            (y_np - self.y_mean) / self.y_scale,
            dtype=torch.double,
            device=self.device,
        )
        self.yvar_train = torch.as_tensor(
            np.maximum(yvar_np / self.y_scale**2, 1.0e-8),
            dtype=torch.double,
            device=self.device,
        )
        pyro.clear_param_store()
        pyro.set_rng_seed(self.seed)
        started = time.perf_counter()
        kernel = NUTS(
            self._pyro_model,
            max_tree_depth=self.nuts.max_tree_depth,
            jit_compile=False,
        )
        mcmc = MCMC(
            kernel,
            warmup_steps=self.nuts.warmup_steps,
            num_samples=self.nuts.num_samples,
            disable_progbar=True,
        )
        mcmc.run()
        raw = mcmc.get_samples()
        step = max(1, self.nuts.thinning)
        self.samples = {key: value[::step].detach() for key, value in raw.items()}
        self.fit_seconds = time.perf_counter() - started

    def _pyro_model(self) -> None:
        assert self.x_train is not None
        assert self.y_train is not None
        assert self.yvar_train is not None
        kwargs = {"dtype": self.x_train.dtype, "device": self.x_train.device}
        tau = pyro.sample(
            "global_inverse_lengthscale",
            pyro.distributions.HalfCauchy(torch.tensor(0.1, **kwargs)),
        )
        inverse_lengthscale = pyro.sample(
            "inverse_lengthscale",
            pyro.distributions.HalfCauchy(
                tau.expand(self.x_train.shape[1])
            ).to_event(1),
        )
        lengthscale = inverse_lengthscale.reciprocal()
        outputscale = pyro.sample(
            "outputscale",
            pyro.distributions.Gamma(
                torch.tensor(2.0, **kwargs), torch.tensor(0.15, **kwargs)
            ),
        )
        covariance = _tree_kernel(
            self.x_train,
            self.x_train,
            self.edges,
            self.latent_dim,
            lengthscale,
            outputscale,
        )
        covariance = covariance + torch.diag(self.yvar_train + 1.0e-6)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                torch.zeros(len(self.x_train), **kwargs),
                covariance_matrix=covariance,
            ),
            obs=self.y_train,
        )

    def component_lcb(
        self,
        x_test: np.ndarray,
        kappa: float,
        chunk_size: int = 128,
    ) -> np.ndarray:
        if self.samples is None or self.x_train is None or self.y_train is None:
            raise RuntimeError("surrogate has not been fitted")
        x_all = torch.as_tensor(x_test, dtype=torch.double, device=self.device)
        scores: list[np.ndarray] = []
        inverse_lengthscale = self.samples["inverse_lengthscale"]
        output = self.samples["outputscale"]
        for start in range(0, len(x_all), chunk_size):
            xt = x_all[start : start + chunk_size]
            edge_mu = torch.zeros((len(self.edges), len(xt)), dtype=torch.double, device=self.device)
            edge_second = torch.zeros_like(edge_mu)
            n_draws = len(inverse_lengthscale)
            for draw in range(n_draws):
                lengthscale = inverse_lengthscale[draw].reciprocal()
                total = _tree_kernel(
                    self.x_train,
                    self.x_train,
                    self.edges,
                    self.latent_dim,
                    lengthscale,
                    output[draw],
                )
                total = total + torch.diag(self.yvar_train + 1.0e-6)
                chol = torch.linalg.cholesky(total)
                alpha = torch.cholesky_solve(self.y_train[:, None], chol).squeeze(-1)
                cross = output[draw] * _tree_kernel_components(
                    xt,
                    self.x_train,
                    self.edges,
                    self.latent_dim,
                    lengthscale,
                )
                mu = torch.matmul(cross, alpha)
                right_hand_side = cross.permute(2, 0, 1).reshape(
                    len(self.x_train), len(self.edges) * len(xt)
                )
                solved = torch.linalg.solve_triangular(
                    chol, right_hand_side, upper=False
                )
                variance = torch.clamp(
                    output[draw]
                    - solved.square().sum(dim=0).reshape(len(self.edges), len(xt)),
                    min=1.0e-10,
                )
                edge_mu += mu / n_draws
                edge_second += (variance + mu.square()) / n_draws
            edge_var = torch.clamp(edge_second - edge_mu.square(), min=1.0e-10)
            standardized = edge_mu.sum(dim=0) - kappa * torch.sqrt(edge_var).sum(dim=0)
            score = self.y_mean + self.y_scale * standardized
            scores.append(score.detach().cpu().numpy())
        return np.concatenate(scores)

    def summary(self) -> dict:
        if self.samples is None:
            return {"status": "unfitted"}
        inv = self.samples["inverse_lengthscale"]
        return {
            "status": "ok",
            "fit_seconds": self.fit_seconds,
            "posterior_samples": int(len(inv)),
            "median_inverse_lengthscale": inv.median(dim=0).values.cpu().tolist(),
            "median_global_inverse_lengthscale": float(
                self.samples["global_inverse_lengthscale"].median().cpu()
            ),
            "edges": [list(edge) for edge in self.edges],
        }


def knn_graph(latent: np.ndarray, k: int) -> tuple[tuple[tuple[int, float], ...], ...]:
    points = np.asarray(latent, dtype=float)
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    graph: list[list[tuple[int, float]]] = [[] for _ in range(len(points))]
    for i in range(len(points)):
        for j in np.argsort(distance[i])[1 : k + 1]:
            graph[i].append((int(j), float(distance[i, j])))
            graph[int(j)].append((i, float(distance[i, j])))
    return tuple(tuple(sorted(set(row))) for row in graph)


def dijkstra_path(
    graph: tuple[tuple[tuple[int, float], ...], ...], source: int, target: int
) -> tuple[int, ...]:
    if source == target:
        return (source,)
    distance = [float("inf")] * len(graph)
    previous = [-1] * len(graph)
    distance[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        current, node = heapq.heappop(queue)
        if current != distance[node]:
            continue
        if node == target:
            break
        for neighbor, weight in graph[node]:
            candidate = current + weight
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    if previous[target] < 0:
        raise RuntimeError("anchor graph is disconnected")
    path = [target]
    while path[-1] != source:
        path.append(previous[path[-1]])
    return tuple(reversed(path))


def design_to_latent(design: Iterable[int], latent: np.ndarray) -> np.ndarray:
    return np.asarray(latent, dtype=float)[np.asarray(tuple(design), dtype=int)].reshape(-1)


@dataclass
class TrustRegion:
    length: float = 0.5
    minimum: float = 0.05
    maximum: float = 1.0
    success_tolerance: int = 3
    failure_tolerance: int = 3
    successes: int = 0
    failures: int = 0

    def update(self, improved: bool) -> None:
        if improved:
            self.successes += 1
            self.failures = 0
            if self.successes >= self.success_tolerance:
                self.length = min(self.maximum, self.length * 1.5)
                self.successes = 0
        else:
            self.failures += 1
            self.successes = 0
            if self.failures >= self.failure_tolerance:
                self.length = max(self.minimum, self.length / 2.0)
                self.failures = 0


def within_trust_region(
    designs: np.ndarray,
    incumbent: np.ndarray,
    latent: np.ndarray,
    length: float,
) -> np.ndarray:
    center = design_to_latent(incumbent, latent)
    values = np.asarray([design_to_latent(row, latent) for row in designs])
    return np.max(np.abs(values - center), axis=1) <= length / 2.0 + 1.0e-12


def graph_evolution_candidates(
    incumbent: np.ndarray,
    latent: np.ndarray,
    graph: tuple[tuple[tuple[int, float], ...], ...],
    trust_length: float,
    n_candidates: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_variables = len(incumbent)
    pool: set[tuple[int, ...]] = {tuple(int(x) for x in incumbent)}
    attempts = 0
    while len(pool) < n_candidates and attempts < n_candidates * 80:
        attempts += 1
        child = np.asarray(incumbent, dtype=int).copy()
        n_mutations = max(1, int(rng.poisson(1.0)))
        for variable in rng.choice(n_variables, size=min(n_variables, n_mutations), replace=False):
            source = int(child[variable])
            target = int(rng.integers(0, len(latent)))
            path = dijkstra_path(graph, source, target)
            step = int(rng.integers(0, len(path)))
            child[variable] = path[step]
        if within_trust_region(child[None, :], incumbent, latent, trust_length)[0]:
            pool.add(tuple(int(x) for x in child))
    if len(pool) < 2:
        for variable in range(n_variables):
            for neighbor, _ in graph[int(incumbent[variable])]:
                child = np.asarray(incumbent, dtype=int).copy()
                child[variable] = neighbor
                if within_trust_region(child[None, :], incumbent, latent, trust_length)[0]:
                    pool.add(tuple(int(x) for x in child))
    return np.asarray(sorted(pool), dtype=int)


def unique_random_designs(
    rng: np.random.Generator,
    n_designs: int,
    n_variables: int,
    n_levels: int,
) -> np.ndarray:
    """Draw unique points from the finite catalog tensor product."""
    capacity = n_levels**n_variables
    if n_designs > capacity:
        raise ValueError("requested more initial designs than the catalog contains")
    designs: set[tuple[int, ...]] = set()
    while len(designs) < n_designs:
        batch = rng.integers(0, n_levels, size=(max(8, n_designs), n_variables))
        designs.update(tuple(int(value) for value in row) for row in batch)
    return np.asarray(sorted(designs)[:n_designs], dtype=int)


def unevaluated_local_candidates(
    incumbent: np.ndarray,
    latent: np.ndarray,
    graph: tuple[tuple[tuple[int, float], ...], ...],
    trust_length: float,
    evaluated: set[tuple[int, ...]],
) -> np.ndarray:
    """Enumerate valid one-edge graph moves without leaving the trust region."""
    pool: set[tuple[int, ...]] = set()
    for variable in range(len(incumbent)):
        for neighbor, _ in graph[int(incumbent[variable])]:
            child = np.asarray(incumbent, dtype=int).copy()
            child[variable] = neighbor
            key = tuple(int(value) for value in child)
            if key in evaluated:
                continue
            if within_trust_region(child[None, :], incumbent, latent, trust_length)[0]:
                pool.add(key)
    if not pool:
        return np.empty((0, len(incumbent)), dtype=int)
    return np.asarray(sorted(pool), dtype=int)


def stable_seed(label: str, seed: int, iteration: int) -> int:
    digest = hashlib.sha256(f"{label}:{seed}:{iteration}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


class PaperCobaltOptimizer:
    """Fixed-budget active-learning loop matching the paper pseudocode."""

    def __init__(
        self,
        latent: np.ndarray,
        evaluator: Callable[[np.ndarray, int], tuple[float, list[float], dict]],
        n_variables: int,
        n_initial: int,
        n_sequential: int,
        kappa: float,
        n_candidates: int,
        nuts: NutsConfig,
        device: torch.device,
        seed: int,
        progress_callback: Callable[[dict], None] | None = None,
        initial_designs: np.ndarray | None = None,
    ) -> None:
        self.latent = np.asarray(latent, dtype=float)
        self.evaluator = evaluator
        self.n_variables = n_variables
        self.n_initial = n_initial
        self.n_sequential = n_sequential
        self.kappa = kappa
        self.n_candidates = n_candidates
        self.nuts = nuts
        self.device = device
        self.seed = seed
        self.progress_callback = progress_callback
        if initial_designs is None:
            self.initial_designs = None
        else:
            initial = np.asarray(initial_designs, dtype=int)
            if initial.shape != (n_initial, n_variables):
                raise ValueError("initial_designs has the wrong shape")
            if np.any(initial < 0) or np.any(initial >= len(self.latent)):
                raise ValueError("initial_designs contains an invalid catalog index")
            if len({tuple(row) for row in initial}) != len(initial):
                raise ValueError("initial_designs must be unique")
            self.initial_designs = initial.copy()
        self.graph = knn_graph(self.latent, min(5, len(self.latent) - 1))

    def run(self) -> dict:
        rng = np.random.default_rng(self.seed)
        initial = (
            self.initial_designs.copy()
            if self.initial_designs is not None
            else unique_random_designs(
                rng,
                self.n_initial,
                self.n_variables,
                len(self.latent),
            )
        )
        history: list[dict] = []
        evaluated: set[tuple[int, ...]] = set()
        for index, design in enumerate(initial):
            self._evaluate(design, -(self.n_initial - index), history, evaluated)
        trust = TrustRegion()
        tree_log: list[dict] = []
        for iteration in range(1, self.n_sequential + 1):
            feasible = [row for row in history if row["feasible"]]
            incumbent_row = min(feasible or history, key=lambda row: (
                0.0 if row["feasible"] else sum(max(c, 0.0) for c in row["constraints"]),
                row["objective"],
            ))
            incumbent = np.asarray(incumbent_row["design"], dtype=int)
            x = np.asarray([design_to_latent(row["design"], self.latent) for row in history])
            y = np.asarray([row["objective"] for row in history])
            yvar = np.asarray([row["objective_observation_variance"] for row in history])
            tree_seed = stable_seed("tree", self.seed, iteration)
            edges = uniform_spanning_tree(self.n_variables, tree_seed)
            model = AdditiveTreeSaasGP(
                self.n_variables,
                self.latent.shape[1],
                edges,
                self.nuts,
                self.device,
                stable_seed("nuts", self.seed, iteration),
            )
            model.fit(x, y, yvar)
            candidates = graph_evolution_candidates(
                incumbent,
                self.latent,
                self.graph,
                trust.length,
                self.n_candidates,
                stable_seed("search", self.seed, iteration),
            )
            candidates = np.asarray(
                [row for row in candidates if tuple(int(v) for v in row) not in evaluated],
                dtype=int,
            )
            search_length = trust.length
            while len(candidates) == 0 and search_length < trust.maximum:
                search_length = min(trust.maximum, search_length * 1.5)
                candidates = graph_evolution_candidates(
                    incumbent,
                    self.latent,
                    self.graph,
                    search_length,
                    self.n_candidates,
                    stable_seed("search-expand", self.seed, iteration + int(1000 * search_length)),
                )
                candidates = np.asarray(
                    [
                        row
                        for row in candidates
                        if tuple(int(v) for v in row) not in evaluated
                    ],
                    dtype=int,
                )
            if len(candidates) == 0:
                candidates = unevaluated_local_candidates(
                    incumbent,
                    self.latent,
                    self.graph,
                    search_length,
                    evaluated,
                )
            if len(candidates) == 0:
                raise RuntimeError(
                    "the maximum graph trust region contains no unevaluated anchored design"
                )
            trust.length = search_length
            candidate_x = np.asarray(
                [design_to_latent(row, self.latent) for row in candidates]
            )
            score = model.component_lcb(candidate_x, self.kappa)
            selected = candidates[int(np.argmin(score))]
            previous_best = min(
                (row["objective"] for row in feasible), default=float("inf")
            )
            row = self._evaluate(selected, iteration, history, evaluated)
            improved = row["feasible"] and row["objective"] < previous_best
            trust.update(improved)
            tree_log.append(
                {
                    "iteration": iteration,
                    "tree_seed": tree_seed,
                    "edges": [list(edge) for edge in edges],
                    "trust_region_length": trust.length,
                    "surrogate": model.summary(),
                    "candidate_count": int(len(candidates)),
                    "selected_lcb": float(np.min(score)),
                }
            )
        feasible = [row for row in history if row["feasible"]]
        recommendation = min(feasible, key=lambda row: row["objective"]) if feasible else None
        return {
            "history": history,
            "tree_log": tree_log,
            "recommendation": recommendation,
            "n_evaluations": len(history),
        }

    def _evaluate(
        self,
        design: np.ndarray,
        iteration: int,
        history: list[dict],
        evaluated: set[tuple[int, ...]],
    ) -> dict:
        design = np.asarray(design, dtype=int)
        key = tuple(int(value) for value in design)
        objective, constraints, stats = self.evaluator(
            design, stable_seed("mc", self.seed, iteration)
        )
        variance = max(float(stats["objective_observation_variance"]), 1.0e-12)
        row = {
            "iteration": iteration,
            "design": list(key),
            "objective": float(objective),
            "constraints": [float(value) for value in constraints],
            "feasible": bool(all(value <= 0.0 for value in constraints)),
            "objective_observation_variance": variance,
            "mc": stats,
        }
        history.append(row)
        evaluated.add(key)
        if self.progress_callback is not None:
            self.progress_callback({
                "history": history,
                "n_evaluations": len(history),
                "latest": row,
            })
        return row
