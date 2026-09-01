"""Nominal I-section catalogue shared by the planar truss benchmarks."""

from __future__ import annotations

import numpy as np


def i_section_properties(b: float, h: float, tf: float, tw: float) -> np.ndarray:
    """Return ``[area, strong_axis_inertia, weak_axis_inertia]`` in SI units."""
    web_height = h - 2.0 * tf
    if min(b, h, tf, tw, web_height) <= 0.0:
        raise ValueError("invalid I-section dimensions")
    area = 2.0 * b * tf + tw * web_height
    strong_axis = (b * h**3 - (b - tw) * web_height**3) / 12.0
    weak_axis = 2.0 * tf * b**3 / 12.0 + web_height * tw**3 / 12.0
    return np.array([area, strong_axis, weak_axis], dtype=float)


def build_54_section_catalog() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build the released 54 nominal I-sections and their physical descriptors.

    The catalogue is a deterministic 6-by-9 grid of manufacturable I-section
    dimensions. It is a benchmark catalogue, not a claim about commercial
    availability of any particular profile designation.
    """
    flange_widths = np.linspace(0.050, 0.100, 6)
    total_depths = np.linspace(0.060, 0.180, 9)
    shapes: list[tuple[float, float, float, float]] = []
    names: list[str] = []
    for i, b in enumerate(flange_widths):
        for j, h in enumerate(total_depths):
            tf = 0.0040 + 0.0003 * i
            tw = 0.0035 + 0.00025 * j
            shapes.append((float(b), float(h), float(tf), float(tw)))
            names.append(f"I-{len(shapes):02d}")
    shape_array = np.asarray(shapes, dtype=float)
    descriptors = np.vstack([i_section_properties(*shape) for shape in shape_array])
    return descriptors, shape_array, tuple(names)


def sample_section_properties(
    nominal_shapes: np.ndarray,
    rng: np.random.Generator,
    shape_cov: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample independent normal shape errors and return shapes and descriptors."""
    shapes = np.asarray(nominal_shapes, dtype=float)
    realised = rng.normal(loc=shapes, scale=shape_cov * shapes)
    b, h, tf, tw = realised.T
    valid = (b > 0.0) & (h > 0.0) & (tf > 0.0) & (tw > 0.0) & (h > 2.0 * tf)
    if not np.all(valid):
        raise RuntimeError("normal shape sample produced a nonphysical section")
    properties = np.vstack([
        i_section_properties(bi, hi, tfi, twi)
        for bi, hi, tfi, twi in realised
    ])
    return realised, properties
