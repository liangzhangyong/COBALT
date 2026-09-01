"""Finite-element oracle and uncertainty sampler for the 105-member planar truss."""

from __future__ import annotations

import numpy as np

from section_catalog import build_54_section_catalog, sample_section_properties


YOUNGS_MODULUS = 2.10e11
DENSITY = 7850.0
GRAVITY = 9.81
NOMINAL_LOAD = 5000.0
MASS_LIMIT = 5000.0
SHAPE_COV = 0.05
MATERIAL_COV = 0.05
LOAD_COV = 0.01
GEOMETRY_RELATIVE_STD = 0.001
CATALOG_PROPERTIES, CATALOG_SHAPES, CATALOG_LABELS = build_54_section_catalog()


def _topology() -> tuple[np.ndarray, np.ndarray]:
    """Build the 6-by-6-node, 105-member planar truss."""
    nx = ny = 6
    def node(i: int, j: int) -> int:
        return i + j * nx
    coordinates = np.array([[3.0 * i, 3.0 * j, 0.0] for j in range(ny) for i in range(nx)], dtype=float)
    elements: list[tuple[int, int, int]] = []
    for j in range(ny):
        for i in range(nx - 1):
            elements.append((node(i, j), node(i + 1, j), 0))
    for i in range(1, nx):
        for j in range(ny - 1):
            elements.append((node(i, j), node(i, j + 1), 1))
    for j in range(ny - 1):
        for i in range(nx - 1):
            elements.append((node(i, j), node(i + 1, j + 1), 2))
            elements.append((node(i, j + 1), node(i + 1, j), 3))
    element_array = np.asarray(elements, dtype=int)
    if len(element_array) != 105:
        raise RuntimeError(f"expected 105 members, got {len(element_array)}")
    return coordinates, element_array


NODE_COORDINATES, ELEMENTS = _topology()


def build_105beam_section_catalog(n_sections: int = 54) -> tuple[np.ndarray, np.ndarray]:
    """Return the requested prefix of the released 54-section catalogue."""
    if not 2 <= n_sections <= len(CATALOG_PROPERTIES):
        raise ValueError(f"n_sections must be in [2, {len(CATALOG_PROPERTIES)}]")
    return CATALOG_PROPERTIES[:n_sections].copy(), np.zeros(n_sections, dtype=int)


def get_topology() -> dict:
    """Return the topology, loading, supports, and material constants."""
    return {
        "n_nodes": 36,
        "n_elements": 105,
        "n_design_variables": 105,
        "node_coordinates": NODE_COORDINATES.copy(),
        "elements": ELEMENTS.copy(),
        "supports": [0, 6, 12, 18, 24, 30],
        "load_nodes": [5],
        "load_mag_N": NOMINAL_LOAD,
        "mass_limit_kg": MASS_LIMIT,
        "material": {"youngs_modulus_Pa": YOUNGS_MODULUS, "density_kg_m3": DENSITY},
    }


def sample_uncertainty(section_indices: np.ndarray, rng: np.random.Generator) -> dict:
    """Draw four independent normal uncertainty sources for one MC realisation."""
    indices = np.asarray(section_indices, dtype=int)
    if indices.shape != (105,) or np.any(indices < 0) or np.any(indices >= len(CATALOG_SHAPES)):
        raise ValueError("105-beam design must contain 105 catalogue indices")
    shapes, properties = sample_section_properties(CATALOG_SHAPES[indices], rng, SHAPE_COV)
    nominal_lengths = _nominal_lengths()
    # Each member has an independent tolerance L_i ~ N(mu_L_i,
    # (0.001 mu_L_i)^2), including the diagonal members.
    effective_lengths = rng.normal(
        nominal_lengths, GEOMETRY_RELATIVE_STD * nominal_lengths
    )
    young_modulus = float(rng.normal(YOUNGS_MODULUS, MATERIAL_COV * YOUNGS_MODULUS))
    load = float(rng.normal(NOMINAL_LOAD, LOAD_COV * NOMINAL_LOAD))
    if np.any(effective_lengths <= 0.0) or young_modulus <= 0.0 or load <= 0.0:
        raise RuntimeError("normal uncertainty sample was nonphysical")
    return {
        "section_shapes": shapes,
        "section_properties": properties,
        "youngs_modulus": young_modulus,
        "load": load,
        "element_lengths": effective_lengths,
    }


def evaluate_105beam(
    design: np.ndarray,
    *,
    young_modulus: float = YOUNGS_MODULUS,
    load: float = NOMINAL_LOAD,
    element_lengths: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate one deterministic 105-member realisation.

    ``design`` is either 105 catalogue indices or 315 interleaved values
    ``[A_1, I_y1, I_z1, …]``. Geometry errors alter element lengths in stiffness,
    mass, and buckling capacity while the nominal node layout remains anchored.
    """
    design = np.asarray(design, dtype=float)
    if design.size == 105:
        indices = design.astype(int)
        if np.any(indices < 0) or np.any(indices >= len(CATALOG_PROPERTIES)):
            raise ValueError("catalogue index out of range")
        properties = CATALOG_PROPERTIES[indices]
    elif design.size == 315:
        properties = design.reshape(105, 3)
    else:
        raise ValueError("design must contain 105 indices or 315 physical values")
    area, iy, iz = properties.T
    if np.any(properties <= 0.0) or young_modulus <= 0.0 or load <= 0.0:
        raise ValueError("section properties, material, and load must be positive")
    nominal_lengths = _nominal_lengths()
    effective_lengths = nominal_lengths if element_lengths is None else np.asarray(element_lengths, dtype=float)
    if effective_lengths.shape != (105,) or np.any(effective_lengths <= 0.0):
        raise ValueError("element_lengths must contain 105 positive values")
    youngs = np.full(105, float(young_modulus))
    mass, nodal_weight = _mass_and_self_weight(area, effective_lengths)
    gdof = 3 * len(NODE_COORDINATES)
    force = np.zeros(gdof)
    force[3 * 5 + 1] = -float(load)
    for node, weight in enumerate(nodal_weight):
        force[3 * node + 1] -= weight
    stiffness, element_data = _stiffness(youngs, area, effective_lengths)
    supports = [0, 6, 12, 18, 24, 30]
    prescribed = {3 * node + axis for node in supports for axis in (0, 1, 2)}
    prescribed.update(3 * node + 2 for node in range(len(NODE_COORDINATES)))
    active = np.setdiff1d(np.arange(gdof), sorted(prescribed))
    displacement = np.zeros(gdof)
    displacement[active] = np.linalg.solve(stiffness[np.ix_(active, active)], force[active])
    energy = 0.5 * float(force @ displacement)
    buckling, max_displacement = _buckling(youngs, area, iy, iz, displacement, nominal_lengths, effective_lengths)
    d_energy = _energy_sensitivity(displacement, element_data)
    d_mass = DENSITY * effective_lengths
    d_buckling = np.zeros((2, 105))
    return energy, mass, buckling, d_energy, d_mass, d_buckling, max_displacement


def _nominal_lengths() -> np.ndarray:
    return np.array([np.linalg.norm(NODE_COORDINATES[i] - NODE_COORDINATES[j]) for i, j, _ in ELEMENTS])


def _mass_and_self_weight(area: np.ndarray, lengths: np.ndarray) -> tuple[float, np.ndarray]:
    masses = DENSITY * area * lengths
    nodal_weight = np.zeros(len(NODE_COORDINATES))
    for mass, (i, j, _) in zip(masses, ELEMENTS):
        nodal_weight[i] += 0.5 * GRAVITY * mass
        nodal_weight[j] += 0.5 * GRAVITY * mass
    return float(masses.sum()), nodal_weight


def _stiffness(youngs: np.ndarray, area: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, list[int]]]]:
    gdof = 3 * len(NODE_COORDINATES)
    stiffness = np.zeros((gdof, gdof))
    data: list[tuple[np.ndarray, list[int]]] = []
    for e, (i, j, _) in enumerate(ELEMENTS):
        direction = NODE_COORDINATES[j] - NODE_COORDINATES[i]
        direction /= np.linalg.norm(direction)
        transform = np.outer(direction, direction)
        element_stiffness = youngs[e] * area[e] / lengths[e] * np.block([[transform, -transform], [-transform, transform]])
        dof = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        stiffness[np.ix_(dof, dof)] += element_stiffness
        data.append((element_stiffness, dof))
    return stiffness, data


def _buckling(youngs: np.ndarray, area: np.ndarray, iy: np.ndarray, iz: np.ndarray, displacement: np.ndarray, nominal_lengths: np.ndarray, effective_lengths: np.ndarray) -> tuple[np.ndarray, float]:
    margins = np.empty((2, 105))
    for e, (i, j, _) in enumerate(ELEMENTS):
        dof = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        deformed = np.concatenate([NODE_COORDINATES[i], NODE_COORDINATES[j]]) + displacement[dof]
        current_length = np.linalg.norm(deformed[3:] - deformed[:3])
        axial_force = youngs[e] * area[e] * (current_length - nominal_lengths[e]) / nominal_lengths[e]
        critical_y = -np.pi**2 * youngs[e] * iy[e] / effective_lengths[e]**2
        critical_z = -np.pi**2 * youngs[e] * iz[e] / effective_lengths[e]**2
        margins[:, e] = [critical_y - axial_force, critical_z - axial_force]
    max_displacement = float(np.max(np.linalg.norm(displacement.reshape(-1, 3), axis=1)))
    return np.max(margins, axis=1), max_displacement


def _energy_sensitivity(displacement: np.ndarray, element_data: list[tuple[np.ndarray, list[int]]]) -> np.ndarray:
    sensitivity = np.zeros(105)
    for e, (stiffness, dof) in enumerate(element_data):
        sensitivity[e] = -0.5 * displacement[dof] @ (stiffness / max(stiffness.max(), 1.0e-16)) @ displacement[dof]
    return sensitivity


if __name__ == "__main__":
    topology = get_topology()
    result = evaluate_105beam(np.zeros(105, dtype=int))
    assert topology["n_nodes"] == 36 and topology["n_elements"] == 105
    assert np.all(np.isfinite([result[0], result[1], *result[2]]))
    print("105-beam topology and FEA smoke test passed.")
