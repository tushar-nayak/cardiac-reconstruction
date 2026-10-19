import unittest

import numpy as np

from cardiac_common.geometry import geometry_comparison, mask_geometry, mesh_geometry


def tetrahedron():
    vertices = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    return vertices, faces


class GeometryTests(unittest.TestCase):
    def test_closed_sphere_topology_integrates_to_four_pi(self):
        vertices, faces = tetrahedron()
        report = mesh_geometry(vertices, faces)
        self.assertEqual(report["status"], "closed_surface")
        self.assertEqual(report["euler_characteristic"], 2)
        self.assertEqual(report["connected_components"], 1)
        self.assertAlmostEqual(report["integrated_gaussian_curvature"], 4 * np.pi)
        self.assertAlmostEqual(report["gauss_bonnet_residual"], 0)

    def test_two_components_and_unused_vertices(self):
        vertices, faces = tetrahedron()
        report = mesh_geometry(np.vstack((vertices, vertices + 10, [[100, 100, 100]])),
                               np.vstack((faces, faces + 4)))
        self.assertEqual(report["vertices"], 8)
        self.assertEqual(report["connected_components"], 2)
        self.assertEqual(report["euler_characteristic"], 4)
        self.assertAlmostEqual(report["integrated_gaussian_curvature"], 8 * np.pi)

    def test_torus_integrates_to_zero(self):
        n, m = 12, 8
        u, v = np.meshgrid(np.arange(n) * 2 * np.pi / n,
                           np.arange(m) * 2 * np.pi / m, indexing="ij")
        vertices = np.stack(((2 + np.cos(v)) * np.cos(u),
                             (2 + np.cos(v)) * np.sin(u), np.sin(v)), axis=-1).reshape(-1, 3)
        faces = []
        for i in range(n):
            for j in range(m):
                a, b = i * m + j, ((i + 1) % n) * m + j
                c, d = ((i + 1) % n) * m + (j + 1) % m, i * m + (j + 1) % m
                faces.extend(((a, b, c), (a, c, d)))
        report = mesh_geometry(vertices, np.asarray(faces))
        self.assertEqual(report["status"], "closed_surface")
        self.assertEqual(report["euler_characteristic"], 0)
        self.assertAlmostEqual(report["integrated_gaussian_curvature"], 0)

    def test_open_triangle_has_boundary_term_but_no_closed_integral(self):
        report = mesh_geometry(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]))
        self.assertEqual(report["status"], "open_surface")
        self.assertEqual(report["boundary_edges"], 3)
        self.assertIsNone(report["integrated_gaussian_curvature"])
        self.assertAlmostEqual(report["boundary_turning_sum"], 2 * np.pi)
        self.assertAlmostEqual(report["gauss_bonnet_residual"], 0)

    def test_vertex_touching_closed_components_are_nonmanifold(self):
        vertices, faces = tetrahedron()
        vertices = np.vstack((vertices, -vertices[1:] + 2 * vertices[0]))
        second = np.array([0, 4, 5, 6])[faces]
        report = mesh_geometry(vertices, np.vstack((faces, second)))
        self.assertEqual(report["nonmanifold_edges"], 0)
        self.assertEqual(report["nonmanifold_vertices"], 1)
        self.assertEqual(report["status"], "nonmanifold_surface")
        self.assertIsNone(report["integrated_gaussian_curvature"])

    def test_degenerate_or_duplicate_faces_do_not_report_curvature(self):
        vertices, faces = tetrahedron()
        for invalid in (np.vstack((faces, faces[:1])), np.array([[0, 0, 1]])):
            report = mesh_geometry(vertices, invalid)
            self.assertEqual(report["status"], "invalid_faces")
            self.assertIsNone(report["integrated_gaussian_curvature"])

    def test_integral_is_unchanged_by_shape_deformation(self):
        vertices, faces = tetrahedron()
        before = mesh_geometry(vertices, faces)
        after = mesh_geometry(vertices * [2, 3, 4], faces)
        self.assertNotEqual(before["surface_area_mm2"], after["surface_area_mm2"])
        self.assertAlmostEqual(before["integrated_gaussian_curvature"], after["integrated_gaussian_curvature"])

    def test_delta_k_sign_and_component_change(self):
        target = np.zeros((20, 20, 20), dtype=bool)
        target[3:7, 3:7, 3:7] = True
        prediction = target.copy()
        prediction[12:16, 12:16, 12:16] = True
        report = geometry_comparison(prediction, target, np.eye(4))
        self.assertEqual(report["delta_k_status"], "valid")
        self.assertAlmostEqual(report["delta_integrated_gaussian_curvature"], 4 * np.pi)
        self.assertAlmostEqual(report["absolute_delta_integrated_gaussian_curvature"], 4 * np.pi)

    def test_grid_boundary_is_not_silently_capped(self):
        target = np.zeros((8, 8, 8), dtype=bool)
        target[:4, 2:6, 2:6] = True
        report = geometry_comparison(target, target, np.eye(4))
        self.assertEqual(report["prediction"]["status"], "open_surface")
        self.assertTrue(report["prediction"]["foreground_touches_grid_boundary"])
        self.assertIsNone(report["delta_integrated_gaussian_curvature"])

    def test_constant_masks_have_no_isosurface(self):
        for fill in (False, True):
            report = mask_geometry(np.full((4, 4, 4), fill), np.eye(4))
            self.assertEqual(report["status"], "no_isosurface")
            self.assertIsNone(report["integrated_gaussian_curvature"])


if __name__ == "__main__":
    unittest.main()
