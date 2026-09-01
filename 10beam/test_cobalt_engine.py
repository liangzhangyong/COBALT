"""Fast invariant tests for the paper-aligned COBALT optimizer."""

import numpy as np
import torch

from cobalt_engine import (
    AdditiveTreeSaasGP,
    NutsConfig,
    _edge_dims,
    _matern52,
    _tree_kernel,
    graph_evolution_candidates,
    knn_graph,
    uniform_spanning_tree,
    validate_tree,
    within_trust_region,
)


def test_random_tree_invariants() -> None:
    for n_variables in (4, 105):
        first = uniform_spanning_tree(n_variables, 17)
        second = uniform_spanning_tree(n_variables, 17)
        assert first == second
        assert len(first) == n_variables - 1
        validate_tree(first, n_variables)


def test_graph_candidates_are_anchored_and_local() -> None:
    latent = np.column_stack((np.linspace(0.0, 1.0, 12), np.linspace(1.0, 0.0, 12)))
    graph = knn_graph(latent, 3)
    incumbent = np.array([5, 5, 5, 5])
    candidates = graph_evolution_candidates(
        incumbent, latent, graph, trust_length=0.6, n_candidates=16, seed=9
    )
    assert candidates.ndim == 2 and candidates.shape[1] == 4
    assert np.all((0 <= candidates) & (candidates < len(latent)))
    assert np.all(within_trust_region(candidates, incumbent, latent, 0.6))


def test_batched_tree_kernel_matches_edge_sum() -> None:
    generator = torch.Generator().manual_seed(23)
    x1 = torch.rand((5, 8), dtype=torch.double, generator=generator)
    x2 = torch.rand((7, 8), dtype=torch.double, generator=generator)
    lengthscale = 0.2 + torch.rand(8, dtype=torch.double, generator=generator)
    edges = uniform_spanning_tree(4, 19)
    expected = torch.zeros((5, 7), dtype=torch.double)
    for edge in edges:
        dims = _edge_dims(edge, 2)
        expected += 1.7 * _matern52(
            x1[:, dims], x2[:, dims], lengthscale[list(dims)]
        )
    actual = _tree_kernel(
        x1, x2, edges, 2, lengthscale, torch.tensor(1.7, dtype=torch.double)
    )
    assert torch.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_additive_saas_nuts_and_lcb() -> None:
    latent = np.array([[0.0, 0.0], [0.4, 0.8], [1.0, 1.0]])
    designs = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
    x = latent[designs].reshape(len(designs), -1)
    model = AdditiveTreeSaasGP(
        n_variables=4,
        latent_dim=2,
        edges=uniform_spanning_tree(4, 3),
        nuts=NutsConfig(warmup_steps=2, num_samples=2, thinning=1, max_tree_depth=1),
        device=torch.device("cpu"),
        seed=4,
    )
    model.fit(x, np.array([3.0, 2.0, 1.0]), np.array([0.1, 0.2, 0.3]))
    score = model.component_lcb(x, kappa=1.8)
    assert score.shape == (3,)
    assert np.all(np.isfinite(score))
    summary = model.summary()
    assert summary["posterior_samples"] == 2
    assert len(summary["edges"]) == 3


if __name__ == "__main__":
    test_random_tree_invariants()
    test_graph_candidates_are_anchored_and_local()
    test_batched_tree_kernel_matches_edge_sum()
    test_additive_saas_nuts_and_lcb()
    print("paper-aligned COBALT invariant tests passed")
