# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Asset-level tests for the deformable dinosaur used by the dino twist example."""

import os
import unittest
from unittest import mock

import numpy as np

from newton.examples.multiphysics.dino_asset import (
    DINO_ASSET_ENV,
    default_asset_dir,
    load_dino_asset,
    visual_surface_normals,
    youngs_to_lame,
)

_EXPECTED_VERTICES = 876
_EXPECTED_TETS = 3168
_EXPECTED_VISUAL_VERTICES = 851
_EXPECTED_VISUAL_FACES = 1130


class TestDinoAsset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_dir = default_asset_dir()
        if not (asset_dir / "final_mesh.mesh").exists():
            raise unittest.SkipTest(f"dino asset bundle not found at {asset_dir}")
        cls.asset = load_dino_asset(asset_dir)

    def test_tet_mesh_matches_exported_topology(self):
        asset = self.asset
        self.assertEqual(asset.vertices.shape, (_EXPECTED_VERTICES, 3))
        self.assertEqual(asset.tet_indices.size, 4 * _EXPECTED_TETS)
        self.assertTrue(np.issubdtype(asset.tet_indices.dtype, np.integer))
        self.assertGreaterEqual(int(asset.tet_indices.min()), 0)
        self.assertLess(int(asset.tet_indices.max()), _EXPECTED_VERTICES)
        self.assertTrue(np.isfinite(asset.vertices).all())

    def test_materials_are_per_tet_lame_parameters(self):
        asset = self.asset
        expected_mu, expected_lambda = youngs_to_lame(5.0e6, 0.49)
        self.assertEqual(asset.k_mu.shape, (_EXPECTED_TETS,))
        self.assertEqual(asset.k_lambda.shape, (_EXPECTED_TETS,))
        self.assertEqual(asset.density.shape, (_EXPECTED_TETS,))
        np.testing.assert_allclose(asset.k_mu, expected_mu, rtol=1.0e-5)
        np.testing.assert_allclose(asset.k_lambda, expected_lambda, rtol=1.0e-5)
        np.testing.assert_allclose(asset.density, 1100.0, rtol=1.0e-6)
        # Lamé conversion round-trips back to the exported moduli.
        mu = np.asarray(expected_mu, dtype=np.float64)
        lam = np.asarray(expected_lambda, dtype=np.float64)
        youngs = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
        poisson = lam / (2.0 * (lam + mu))
        np.testing.assert_allclose(youngs, 5.0e6, rtol=1.0e-6)
        np.testing.assert_allclose(poisson, 0.49, rtol=1.0e-6)

    def test_visual_binding_shape_and_range(self):
        asset = self.asset
        self.assertEqual(asset.visual_vertices.shape, (_EXPECTED_VISUAL_VERTICES, 3))
        self.assertEqual(asset.visual_faces.shape, (_EXPECTED_VISUAL_FACES, 3))
        self.assertEqual(asset.visual_uv.shape, (_EXPECTED_VISUAL_VERTICES, 2))
        self.assertTrue(np.issubdtype(asset.visual_faces.dtype, np.integer))
        self.assertGreaterEqual(int(asset.visual_faces.min()), 0)
        self.assertLess(int(asset.visual_faces.max()), _EXPECTED_VISUAL_VERTICES)
        self.assertGreaterEqual(float(asset.visual_uv.min()), 0.0)
        self.assertLessEqual(float(asset.visual_uv.max()), 1.0)
        self.assertEqual(asset.visual_vertex_to_particle.shape, (_EXPECTED_VISUAL_VERTICES,))
        self.assertGreaterEqual(int(asset.visual_vertex_to_particle.min()), 0)
        self.assertLess(int(asset.visual_vertex_to_particle.max()), _EXPECTED_VERTICES)
        # Seams duplicate surface vertices, so the binding addresses fewer particles.
        self.assertLess(np.unique(asset.visual_vertex_to_particle).size, _EXPECTED_VERTICES)

    def test_visual_surface_normals_are_unit_length(self):
        asset = self.asset
        normals = visual_surface_normals(asset.visual_vertices.astype(np.float64), asset.visual_faces)
        self.assertEqual(normals.shape, asset.visual_vertices.shape)
        self.assertTrue(np.isfinite(normals).all())
        lengths = np.linalg.norm(normals, axis=1)
        # Every vertex used by a face is unit length; unused vertices stay zero.
        self.assertTrue(np.all((np.abs(lengths - 1.0) < 1.0e-5) | (lengths < 1.0e-12)))

    def test_rest_bounds_match_metric_scaling(self):
        asset = self.asset
        extent = asset.bounds_max - asset.bounds_min
        self.assertAlmostEqual(float(extent.max()), 0.18344834287969147, places=6)

    def test_missing_asset_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_dino_asset(default_asset_dir() / "definitely-not-an-asset-directory")

    def test_asset_directory_env_override(self):
        target = str(default_asset_dir())
        with mock.patch.dict(os.environ, {DINO_ASSET_ENV: target}, clear=False):
            self.assertEqual(str(default_asset_dir()), target)


if __name__ == "__main__":
    unittest.main()
