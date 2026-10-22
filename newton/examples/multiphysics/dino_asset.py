# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Asset loading for the deformable rubber dinosaur used by ``example_dino_twist``.

The checked-in asset bundle (``assets/dino``) contains

* ``final_mesh.mesh`` -- the 876-vertex / 3168-tet volumetric mesh in Medit
  (``MeshVersionFormatted 1``) text format,
* ``heterogeneous_params.npz`` -- per-tet Young's modulus, Poisson ratio,
  density and part labels,
* ``appearance/visual_mesh.glb`` -- the exported appearance surface, with the
  same metric frame as the tet mesh, and ``appearance/visual_to_physics.npz``
  -- the exact binding between the 851 visual vertices (1130 faces, with UV
  seams duplicated) and the physics particles,
* ``appearance/albedo.png`` -- the baked base-color texture.

Deforming playback therefore only needs the physics particles: every visual
vertex follows ``particle_q[visual_vertex_to_particle]``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import newton

DINO_ASSET_ENV = "NEWTON_DINO_ASSET_DIR"
"""Environment variable that overrides the dino asset directory."""


def default_asset_dir() -> Path:
    """Return the checked-in dino asset directory (``<repo>/assets/dino``)."""
    asset_dir = os.environ.get(DINO_ASSET_ENV)
    if asset_dir:
        return Path(asset_dir)
    return Path(__file__).resolve().parents[3] / "assets" / "dino"


def youngs_to_lame(youngs_modulus: float, poisson_ratio: float) -> tuple[float, float]:
    """Convert an isotropic Young's modulus / Poisson ratio pair to Lamé parameters."""
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lam = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    return mu, lam


@dataclass
class DinoAsset:
    """Rest geometry, per-tet materials and bound appearance surface of one dino."""

    vertices: np.ndarray
    """Tet-mesh vertices in meters, shape ``(876, 3)``."""
    tet_indices: np.ndarray
    """Flattened tetrahedron indices, shape ``(3168 * 4,)``."""
    k_mu: np.ndarray
    """First Lamé parameter per tetrahedron [Pa]."""
    k_lambda: np.ndarray
    """Second Lamé parameter per tetrahedron [Pa]."""
    density: np.ndarray
    """Density per tetrahedron [kg/m^3]."""
    visual_vertices: np.ndarray
    """Rest positions of the bound visual surface in the same metric frame as ``vertices``."""
    visual_faces: np.ndarray
    """Triangle indices of the visual surface, shape ``(1130, 3)``."""
    visual_uv: np.ndarray
    """Per-visual-vertex texture coordinates, shape ``(851, 2)``."""
    visual_vertex_to_particle: np.ndarray
    """Particle index driven by each visual vertex, shape ``(851,)``."""
    albedo: Path
    """Path to the baked albedo texture."""
    bounds_min: np.ndarray
    """Rest bounding-box minimum of the tet mesh."""
    bounds_max: np.ndarray
    """Rest bounding-box maximum of the tet mesh."""


def load_dino_asset(asset_dir: str | Path | None = None) -> DinoAsset:
    """Load the dino tet mesh, its per-tet material arrays and the bound visual surface."""
    asset_dir = Path(asset_dir) if asset_dir is not None else default_asset_dir()
    mesh_path = asset_dir / "final_mesh.mesh"
    params_path = asset_dir / "heterogeneous_params.npz"
    binding_path = asset_dir / "appearance" / "visual_to_physics.npz"
    albedo_path = asset_dir / "appearance" / "albedo.png"
    for path in (mesh_path, params_path, binding_path, albedo_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing dino asset file '{path}'. Point {DINO_ASSET_ENV} at the asset directory.")

    tet_mesh = newton.TetMesh.create_from_file(str(mesh_path))
    vertices = np.asarray(tet_mesh.vertices, dtype=np.float32)
    tet_indices = np.asarray(tet_mesh.tet_indices, dtype=np.int32)

    with np.load(params_path) as params:
        tet_E_nu = np.asarray(params["tet_E_nu"], dtype=np.float64)
        tet_density = np.asarray(params["tet_density"], dtype=np.float64)
    if tet_E_nu.shape != (len(tet_indices) // 4, 2):
        raise ValueError(f"Unsupported tet_E_nu shape {tet_E_nu.shape} for {len(tet_indices) // 4} tetrahedra")

    mu = np.empty(len(tet_E_nu), dtype=np.float32)
    lam = np.empty(len(tet_E_nu), dtype=np.float32)
    mu[:], lam[:] = youngs_to_lame(tet_E_nu[:, 0], tet_E_nu[:, 1])

    with np.load(binding_path) as binding:
        visual_vertices = np.asarray(binding["visual_rest_vertices_m"], dtype=np.float32)
        visual_faces = np.asarray(binding["visual_faces"], dtype=np.int32)
        visual_uv = np.asarray(binding["visual_uv"], dtype=np.float32)
        visual_vertex_to_particle = np.asarray(binding["physics_vertex_indices"], dtype=np.int32)
    if visual_vertex_to_particle.shape != (len(visual_vertices),):
        raise ValueError("visual_to_physics binding does not match the visual vertex count")

    return DinoAsset(
        vertices=vertices,
        tet_indices=tet_indices,
        k_mu=mu,
        k_lambda=lam,
        density=tet_density.astype(np.float32),
        visual_vertices=visual_vertices,
        visual_faces=visual_faces,
        visual_uv=visual_uv,
        visual_vertex_to_particle=visual_vertex_to_particle,
        albedo=albedo_path,
        bounds_min=vertices.min(axis=0),
        bounds_max=vertices.max(axis=0),
    )


def visual_surface_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Smooth per-vertex normals for the visual surface from its current positions."""
    normals = np.zeros_like(vertices)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(length, 1.0e-12)


__all__ = [
    "DINO_ASSET_ENV",
    "DinoAsset",
    "default_asset_dir",
    "load_dino_asset",
    "visual_surface_normals",
    "youngs_to_lame",
]
