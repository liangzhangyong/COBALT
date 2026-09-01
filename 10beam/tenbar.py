"""Finite-element oracle and uncertainty sampler for the planar 10-beam truss."""

from __future__ import annotations

import numpy as np

from section_catalog import build_54_section_catalog, sample_section_properties


YOUNGS_MODULUS = 2.10e11
DENSITY = 7850.0
GRAVITY = 9.81
NOMINAL_LOAD = 10000.0
MASS_LIMIT = 240.0
SHAPE_COV = 0.05
MATERIAL_COV = 0.05
LOAD_COV = 0.01
GEOMETRY_RELATIVE_STD = 0.001

NODE_COORDINATES = 3.0 * np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
    [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],
])
ELEMENT_NODES = np.array([
    [0, 1, 0], [1, 2, 0], [3, 4, 0], [4, 5, 0], [1, 4, 1],
    [2, 5, 1], [0, 4, 2], [1, 5, 2], [1, 3, 3], [2, 4, 3],
], dtype=int)
CATALOG_PROPERTIES, CATALOG_SHAPES, CATALOG_LABELS = build_54_section_catalog()


def get_topology() -> dict:
    """Return the nominal geometry, loading, supports, and material constants."""
    return {
        "n_nodes": 6,
        "n_elements": 10,
        "n_design_variables": 4,
        "node_coordinates": NODE_COORDINATES.copy(),
        "elements": ELEMENT_NODES.copy(),
        "supports": [0, 3],
        "load_node": 2,
        "load_N": NOMINAL_LOAD,
        "mass_limit_kg": MASS_LIMIT,
        "material": {"youngs_modulus_Pa": YOUNGS_MODULUS, "density_kg_m3": DENSITY},
    }


def sample_uncertainty(section_indices: np.ndarray, rng: np.random.Generator) -> dict:
    """Draw independent normal material, load, geometry, and shape errors."""
    indices = np.asarray(section_indices, dtype=int)
    if indices.shape != (4,) or np.any(indices < 0) or np.any(indices >= len(CATALOG_SHAPES)):
        raise ValueError("10-beam design must contain four catalogue indices")
    shapes, properties = sample_section_properties(CATALOG_SHAPES[indices], rng, SHAPE_COV)
    nominal_lengths = _nominal_lengths()
    # Each member has its own independent manufacturing-length tolerance.
    # This implements L_i ~ N(mu_L_i, (0.001 mu_L_i)^2), rather than using a
    # global span as the standard deviation for every member.
    realised_lengths = rng.normal(
        nominal_lengths, GEOMETRY_RELATIVE_STD * nominal_lengths
    )
    if np.any(realised_lengths <= 0.0):
        raise RuntimeError("normal geometry sample produced a nonphysical member length")
    young_modulus = float(rng.normal(YOUNGS_MODULUS, MATERIAL_COV * YOUNGS_MODULUS))
    load = float(rng.normal(NOMINAL_LOAD, LOAD_COV * NOMINAL_LOAD))
    if young_modulus <= 0.0 or load <= 0.0:
        raise RuntimeError("normal material or load sample was nonphysical")
    return {
        "section_shapes": shapes,
        "section_properties": properties,
        "youngs_modulus": young_modulus,
        "load": load,
        "element_lengths": realised_lengths,
    }


def evaluate_tenbeam(
    group_properties: np.ndarray,
    *,
    young_modulus: float = YOUNGS_MODULUS,
    load: float = NOMINAL_LOAD,
    element_lengths: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate one deterministic realisation of the 10-beam truss.

    ``group_properties`` has shape ``(4, 3)`` and stores ``[A, I_y, I_z]``.
    ``element_lengths`` carries independent manufacturing length tolerances;
    nominal member directions and strain references remain fixed.
    """
    group_properties = np.asarray(group_properties, dtype=float).reshape(4, 3)
    number_nodes = len(NODE_COORDINATES)
    number_elements = len(ELEMENT_NODES)
    nominal_lengths = _nominal_lengths()
    effective_lengths = nominal_lengths if element_lengths is None else np.asarray(element_lengths, dtype=float)
    if effective_lengths.shape != (number_elements,) or np.any(effective_lengths <= 0.0):
        raise ValueError("element_lengths must contain ten positive values")
    groups = ELEMENT_NODES[:, 2]
    area = group_properties[groups, 0]
    iy = group_properties[groups, 1]
    iz = group_properties[groups, 2]
    if np.any(area <= 0.0) or np.any(iy <= 0.0) or np.any(iz <= 0.0) or young_modulus <= 0.0:
        raise ValueError("section properties and Young's modulus must be positive")
    youngs = np.full(number_elements, float(young_modulus))
    mass, self_weight = _mass_and_self_weight(area, effective_lengths)
    gdof = 3 * number_nodes
    force = np.zeros(gdof)
    force[3 * 2 + 1] = -float(load)
    for node, weight in enumerate(self_weight):
        force[3 * node + 1] -= weight
    stiffness, element_data = _stiffness(youngs, area, effective_lengths)
    prescribed = sorted({3 * node + dof for node in range(number_nodes) for dof in [2]} |
                       {0, 1, 2, 9, 10, 11})
    displacement = np.zeros(gdof)
    active = np.setdiff1d(np.arange(gdof), prescribed)
    displacement[active] = np.linalg.solve(stiffness[np.ix_(active, active)], force[active])
    energy = 0.5 * float(force @ displacement)
    buckling, max_displacement = _buckling(youngs, area, iy, iz, displacement, nominal_lengths, effective_lengths)
    d_energy = _energy_sensitivity(displacement, element_data)
    d_mass = _group_mass_sensitivity(effective_lengths)
    d_buckling = np.zeros((2, 12))
    return energy, mass, buckling, d_energy, d_mass, d_buckling, max_displacement


def tenbar(design: np.ndarray):
    """Compatibility wrapper returning the historical constraint convention."""
    energy, mass, buckling, d_energy, d_mass, d_buckling, max_displacement = evaluate_tenbeam(design)
    return energy, mass - MASS_LIMIT, buckling, d_energy, d_mass, d_buckling, max_displacement


def _nominal_lengths() -> np.ndarray:
    return np.array([
        np.linalg.norm(NODE_COORDINATES[i] - NODE_COORDINATES[j])
        for i, j, _ in ELEMENT_NODES
    ])


def _mass_and_self_weight(area: np.ndarray, lengths: np.ndarray) -> tuple[float, np.ndarray]:
    element_masses = DENSITY * area * lengths
    nodal_weight = np.zeros(len(NODE_COORDINATES))
    for mass, (i, j, _) in zip(element_masses, ELEMENT_NODES):
        nodal_weight[i] += 0.5 * GRAVITY * mass
        nodal_weight[j] += 0.5 * GRAVITY * mass
    return float(element_masses.sum()), nodal_weight


def _stiffness(youngs: np.ndarray, area: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, list[int]]]]:
    stiffness = np.zeros((18, 18))
    data: list[tuple[np.ndarray, list[int]]] = []
    for e, (i, j, _) in enumerate(ELEMENT_NODES):
        direction = NODE_COORDINATES[j] - NODE_COORDINATES[i]
        direction /= np.linalg.norm(direction)
        transform = np.outer(direction, direction)
        element_stiffness = youngs[e] * area[e] / lengths[e] * np.block([[transform, -transform], [-transform, transform]])
        dof = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        stiffness[np.ix_(dof, dof)] += element_stiffness
        data.append((element_stiffness, dof))
    return stiffness, data


def _buckling(youngs: np.ndarray, area: np.ndarray, iy: np.ndarray, iz: np.ndarray, displacement: np.ndarray, nominal_lengths: np.ndarray, effective_lengths: np.ndarray) -> tuple[np.ndarray, float]:
    margins = np.empty((2, len(ELEMENT_NODES)))
    for e, (i, j, _) in enumerate(ELEMENT_NODES):
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
    sensitivity = np.zeros(12)
    for element, (stiffness, dof) in enumerate(element_data):
        group = ELEMENT_NODES[element, 2]
        sensitivity[3 * group] += -0.5 * displacement[dof] @ (stiffness / max(1.0e-16, stiffness.max())) @ displacement[dof]
    return sensitivity


def _group_mass_sensitivity(lengths: np.ndarray) -> np.ndarray:
    sensitivity = np.zeros(12)
    for length, (_, _, group) in zip(lengths, ELEMENT_NODES):
        sensitivity[3 * group] += DENSITY * length
    return sensitivity


if __name__ == "__main__":
    topology = get_topology()
    result = evaluate_tenbeam(CATALOG_PROPERTIES[:4])
    assert topology["n_elements"] == 10 and np.all(np.isfinite([result[0], result[1], *result[2]]))
    print("10-beam topology and FEA smoke test passed.")
