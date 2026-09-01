"""Command-line runner for the paper-aligned COBALT benchmark implementations."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.manifold import Isomap, trustworthiness

from cobalt_engine import NutsConfig, PaperCobaltOptimizer, stable_seed


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def bootstrap_robust_variance(
    values: np.ndarray,
    gamma: float,
    replicates: int,
    seed: int,
) -> float:
    """Estimate MC observation variance for mean + gamma times sample SD."""
    rng = np.random.default_rng(seed)
    n = len(values)
    statistics = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 64):
        count = min(64, replicates - start)
        sample = values[rng.integers(0, n, size=(count, n))]
        statistics[start : start + count] = sample.mean(axis=1) + gamma * sample.std(
            axis=1, ddof=1
        )
    variance = float(statistics.var(ddof=1))
    return max(variance, np.finfo(float).eps)


def robust_statistic(values: np.ndarray, gamma: float) -> float:
    return float(values.mean() + gamma * values.std(ddof=1))


class RobustOracle:
    def __init__(
        self,
        benchmark: str,
        n_mc: int,
        gamma: float,
        bootstrap_replicates: int,
    ) -> None:
        self.benchmark = benchmark
        self.n_mc = n_mc
        self.gamma = gamma
        self.bootstrap_replicates = bootstrap_replicates
        if benchmark == "10beam":
            from section_catalog import build_54_section_catalog
            from tenbar import MASS_LIMIT, evaluate_tenbeam, get_topology, sample_uncertainty

            self.catalog = build_54_section_catalog()[0]
            self.mass_limit = MASS_LIMIT
            self.n_variables = 4
            self.topology = get_topology()
            self.sample_uncertainty = sample_uncertainty

            def evaluate(design: np.ndarray, realization: dict):
                return evaluate_tenbeam(
                    realization["section_properties"],
                    young_modulus=realization["youngs_modulus"],
                    load=realization["load"],
                    element_lengths=realization["element_lengths"],
                )

            self.evaluate_realization = evaluate
        elif benchmark == "105beam":
            from beam105 import (
                MASS_LIMIT,
                build_105beam_section_catalog,
                evaluate_105beam,
                get_topology,
                sample_uncertainty,
            )

            self.catalog = build_105beam_section_catalog(54)[0]
            self.mass_limit = MASS_LIMIT
            self.n_variables = 105
            self.topology = get_topology()
            self.sample_uncertainty = sample_uncertainty

            def evaluate(design: np.ndarray, realization: dict):
                return evaluate_105beam(
                    realization["section_properties"].reshape(-1),
                    young_modulus=realization["youngs_modulus"],
                    load=realization["load"],
                    element_lengths=realization["element_lengths"],
                )

            self.evaluate_realization = evaluate
        else:
            raise ValueError(f"unknown benchmark: {benchmark}")

    def evaluate(self, design: np.ndarray, seed: int) -> tuple[float, list[float], dict]:
        rng = np.random.default_rng(seed)
        energy = np.empty(self.n_mc, dtype=float)
        mass = np.empty(self.n_mc, dtype=float)
        buckling_y = np.empty(self.n_mc, dtype=float)
        buckling_z = np.empty(self.n_mc, dtype=float)
        started = time.perf_counter()
        for index in range(self.n_mc):
            realization = self.sample_uncertainty(np.asarray(design, dtype=int), rng)
            result = self.evaluate_realization(design, realization)
            energy[index] = result[0]
            mass[index] = result[1]
            buckling_y[index] = result[2][0]
            buckling_z[index] = result[2][1]
            if not np.all(
                np.isfinite(
                    [energy[index], mass[index], buckling_y[index], buckling_z[index]]
                )
            ):
                raise RuntimeError(f"non-finite MC-FEA response at sample {index}")
        objective = robust_statistic(energy, self.gamma)
        constraints = [
            robust_statistic(mass, self.gamma) - self.mass_limit,
            robust_statistic(buckling_y, self.gamma),
            robust_statistic(buckling_z, self.gamma),
        ]
        variance = bootstrap_robust_variance(
            energy,
            self.gamma,
            self.bootstrap_replicates,
            stable_seed("bootstrap", seed, self.n_mc),
        )
        return objective, constraints, {
            "status": "ok",
            "seed": seed,
            "n_mc": self.n_mc,
            "sample_standard_deviation_ddof": 1,
            "objective_observation_variance": variance,
            "uncertainty_sources": [
                "I-section shape b, h, tf, tw: independent normal, CoV 0.05",
                "Young's modulus: normal, CoV 0.05",
                "external load: normal, CoV 0.01",
                "member length: independent normal, CoV 0.001",
            ],
            "energy_mean": float(energy.mean()),
            "energy_std": float(energy.std(ddof=1)),
            "mass_mean": float(mass.mean()),
            "mass_std": float(mass.std(ddof=1)),
            "buckling_y_mean": float(buckling_y.mean()),
            "buckling_y_std": float(buckling_y.std(ddof=1)),
            "buckling_z_mean": float(buckling_z.mean()),
            "buckling_z_std": float(buckling_z.std(ddof=1)),
            "elapsed_seconds": time.perf_counter() - started,
        }


def embedding(catalog: np.ndarray, k: int, m: int) -> tuple[np.ndarray, dict]:
    scale = np.ptp(catalog, axis=0)
    scale[scale == 0.0] = 1.0
    descriptors = (catalog - catalog.min(axis=0)) / scale
    started = time.perf_counter()
    model = Isomap(n_neighbors=k, n_components=m)
    raw = model.fit_transform(descriptors)
    latent_scale = np.ptp(raw, axis=0)
    latent_scale[latent_scale == 0.0] = 1.0
    latent = (raw - raw.min(axis=0)) / latent_scale
    reconstruction = float(model.reconstruction_error())
    return latent, {
        "method": "Isomap",
        "k_iso": k,
        "latent_dimension": m,
        "normalization": "component-wise min-max before and after Isomap",
        "reconstruction_error": reconstruction,
        "neighbor_preservation": float(trustworthiness(descriptors, raw, n_neighbors=k)),
        "fit_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("10beam", "105beam"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--n-mc", type=int, default=500)
    parser.add_argument("--n-validation", type=int, default=3000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--kappa", type=float, default=1.8)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=128)
    parser.add_argument("--nuts-warmup", type=int, default=64)
    parser.add_argument("--nuts-samples", type=int, default=32)
    parser.add_argument("--nuts-thinning", type=int, default=2)
    parser.add_argument("--nuts-max-tree-depth", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    defaults = {"10beam": (15, 185), "105beam": (54, 30)}
    n_initial, n_sequential = defaults[args.benchmark]
    n_initial = args.n_init if args.n_init is not None else n_initial
    n_sequential = args.n_iter if args.n_iter is not None else n_sequential
    k_iso = args.k if args.k is not None else {"10beam": 5, "105beam": 7}[args.benchmark]
    if args.n_mc < 2 or args.n_validation < 2 or args.bootstrap_replicates < 2:
        raise ValueError("MC, validation, and bootstrap sample counts must be at least two")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    total_started = time.perf_counter()
    oracle = RobustOracle(
        args.benchmark, args.n_mc, args.gamma, args.bootstrap_replicates
    )
    latent, embedding_record = embedding(oracle.catalog, k_iso, args.m)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config.update({"n_initial": n_initial, "n_sequential": n_sequential, "k_iso": k_iso})
    payload = {
        "schema_version": 2,
        "status": "running",
        "implementation": "paper-aligned COBALT",
        "run_id": f"cobalt_paper_{args.benchmark}_seed{args.seed}",
        "config": config,
        "protocol": {
            "catalogue_size": int(len(oracle.catalog)),
            "design_variables": oracle.n_variables,
            "objective": "sample mean strain energy + gamma times sample standard deviation",
            "constraints": "same robust statistic for mass and two buckling margins",
            "uncertainty_independence": "all four source classes are mutually independent",
            "mc_failure_policy": "abort candidate and run; no failed sample is discarded",
            "surrogate": "random-tree additive ARD Matern-5/2 SAAS-GP with NUTS marginalization",
            "acquisition": "edge-factorized LCB minimized on graph anchors inside an L-infinity trust region",
            "validation": "disjoint independent Monte Carlo stream",
            "initialization": (
                "all 54 homogeneous catalog assignments"
                if args.benchmark == "105beam" and n_initial == 54
                else "unique seeded catalog assignments"
            ),
        },
        "topology": oracle.topology,
        "embedding": embedding_record,
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "pyro": __import__("pyro").__version__,
            "device": str(device),
        },
    }

    def checkpoint(progress: dict) -> None:
        payload["progress"] = progress
        payload["timing"] = {"total_so_far": time.perf_counter() - total_started}
        atomic_json(args.output, payload)

    try:
        initial_designs = None
        if args.benchmark == "105beam" and n_initial == 54:
            initial_designs = np.repeat(
                np.arange(54, dtype=int)[:, None], oracle.n_variables, axis=1
            )
        optimizer = PaperCobaltOptimizer(
            latent=latent,
            evaluator=oracle.evaluate,
            n_variables=oracle.n_variables,
            n_initial=n_initial,
            n_sequential=n_sequential,
            kappa=args.kappa,
            n_candidates=args.candidates,
            nuts=NutsConfig(
                warmup_steps=args.nuts_warmup,
                num_samples=args.nuts_samples,
                thinning=args.nuts_thinning,
                max_tree_depth=args.nuts_max_tree_depth,
            ),
            device=device,
            seed=args.seed,
            progress_callback=checkpoint,
            initial_designs=initial_designs,
        )
        result = optimizer.run()
        recommendation = result["recommendation"]
        validation_started = time.perf_counter()
        if recommendation is None:
            validation = {"status": "no_feasible_recommendation"}
        else:
            validation_oracle = RobustOracle(
                args.benchmark,
                args.n_validation,
                args.gamma,
                args.bootstrap_replicates,
            )
            validation_seed = stable_seed("independent-validation", args.seed, 0)
            objective, constraints, statistics = validation_oracle.evaluate(
                np.asarray(recommendation["design"], dtype=int), validation_seed
            )
            validation = {
                "status": "ok",
                "seed": validation_seed,
                "n_mc": args.n_validation,
                "design": recommendation["design"],
                "section_indices": recommendation["design"],
                "robust_objective": objective,
                "robust_constraints": constraints,
                "robust_feasible": bool(all(value <= 0.0 for value in constraints)),
                "gmax": max(0.0, max(constraints)),
                "statistics": statistics,
            }
        payload.pop("progress", None)
        payload.update(
            {
                "status": "ok",
                "result": result,
                "independent_validation": validation,
                "cost": {
                    "optimization_design_evaluations": result["n_evaluations"],
                    "optimization_mc_fea_calls": result["n_evaluations"] * args.n_mc,
                    "validation_mc_fea_calls": (
                        args.n_validation if recommendation is not None else 0
                    ),
                    "total_mc_fea_calls": result["n_evaluations"] * args.n_mc
                    + (args.n_validation if recommendation is not None else 0),
                },
                "timing": {
                    "validation": time.perf_counter() - validation_started,
                    "total": time.perf_counter() - total_started,
                },
                "resource": {
                    "end_rss_bytes": process.memory_info().rss,
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                    "torch_peak_allocated_bytes": (
                        torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
                    ),
                },
            }
        )
    except Exception as error:
        payload.update(
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "timing": {"total": time.perf_counter() - total_started},
            }
        )
        atomic_json(args.output, payload)
        raise
    atomic_json(args.output, payload)
    print(json.dumps({"status": "ok", "output": str(args.output), "validation": validation}))


if __name__ == "__main__":
    main()
