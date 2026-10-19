"""Discrete Gaussian curvature on physical-space triangle meshes.

Closed manifold vertices contribute angle defect 2*pi - sum(incident angles).
For open manifolds, boundary turning uses pi - sum(incident angles); the
closed-surface integral and prediction/target delta are deliberately undefined.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from code_gaussian.src.metrics import volume_to_mesh


def mesh_geometry(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("vertices and faces must have shape (N,3) and (F,3)")
    if not np.isfinite(vertices).all() or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("vertices must be finite and faces must contain integer indices")
    result = {
        "status": "no_isosurface", "vertices": 0, "edges": 0, "faces": len(faces),
        "connected_components": 0, "euler_characteristic": None,
        "boundary_edges": 0, "nonmanifold_edges": 0, "nonmanifold_vertices": 0,
        "degenerate_faces": 0, "duplicate_faces": 0, "surface_area_mm2": 0.0,
        "integrated_gaussian_curvature": None,
        "interior_angle_defect_sum": None, "boundary_turning_sum": None,
        "gauss_bonnet_residual": None,
    }
    if not len(faces):
        return result
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("face index outside vertex array")
    # Ignore unreferenced vertices when counting topology.
    used, inverse = np.unique(faces, return_inverse=True)
    vertices, faces = vertices[used], inverse.reshape(-1, 3)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    duplicate_faces = len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))

    # Each edge occurrence carries its two incident face-corner identifiers.
    pairs = np.array([[0, 1], [1, 2], [2, 0]])
    occurrences = faces[:, pairs].reshape(-1, 2)
    corners = np.arange(faces.size).reshape(-1, 3)[:, pairs].reshape(-1, 2)
    sorting = np.argsort(occurrences, axis=1)
    occurrences = np.take_along_axis(occurrences, sorting, axis=1)
    corners = np.take_along_axis(corners, sorting, axis=1)
    edges, edge_inverse, counts = np.unique(occurrences, axis=0, return_inverse=True, return_counts=True)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(len(vertices), len(vertices))).tocsr()
    components = connected_components(graph, directed=False, return_labels=False)
    boundary = np.zeros(len(vertices), dtype=bool)
    boundary[edges[counts == 1].reshape(-1)] = True

    # The faces around each manifold vertex must form one connected fan.
    # Edge counts alone miss two otherwise valid surfaces meeting at one vertex.
    order = np.argsort(edge_inverse)
    starts = np.cumsum(counts) - counts
    shared = starts[counts == 2]
    left = corners[order[shared]].reshape(-1)
    right = corners[order[shared + 1]].reshape(-1)
    corner_graph = coo_matrix((np.ones(len(left)), (left, right)),
                              shape=(faces.size, faces.size)).tocsr()
    _, corner_labels = connected_components(corner_graph, directed=False)
    fans = np.unique(np.column_stack((faces.reshape(-1), corner_labels)), axis=0)
    fan_count = np.bincount(fans[:, 0], minlength=len(vertices))

    result.update(
        vertices=len(vertices), edges=len(edges), connected_components=int(components),
        euler_characteristic=int(len(vertices) - len(edges) + len(faces)),
        boundary_edges=int((counts == 1).sum()), nonmanifold_edges=int((counts > 2).sum()),
        nonmanifold_vertices=int((fan_count > 1).sum()),
        degenerate_faces=int((twice_area == 0).sum()), duplicate_faces=int(duplicate_faces),
        surface_area_mm2=float(twice_area.sum() / 2),
    )
    if result["degenerate_faces"] or result["duplicate_faces"]:
        result["status"] = "invalid_faces"
        return result
    if result["nonmanifold_edges"] or result["nonmanifold_vertices"]:
        result["status"] = "nonmanifold_surface"
        return result

    angle_sums = np.zeros(len(vertices))
    for corner in range(3):
        first = triangles[:, (corner + 1) % 3] - triangles[:, corner]
        second = triangles[:, (corner + 2) % 3] - triangles[:, corner]
        angles = np.arctan2(np.linalg.norm(np.cross(first, second), axis=1),
                            np.einsum("ij,ij->i", first, second))
        np.add.at(angle_sums, faces[:, corner], angles)
    interior = float((2 * np.pi - angle_sums[~boundary]).sum())
    boundary_turning = float((np.pi - angle_sums[boundary]).sum())
    result.update(
        status="open_surface" if boundary.any() else "closed_surface",
        integrated_gaussian_curvature=None if boundary.any() else interior,
        interior_angle_defect_sum=interior, boundary_turning_sum=boundary_turning,
        gauss_bonnet_residual=float(interior + boundary_turning -
                                   2 * np.pi * result["euler_characteristic"]),
    )
    return result


def mask_geometry(mask, affine):
    """Extract the thresholded mask boundary without padding/sealing the grid."""
    mask = np.asarray(mask, dtype=bool)
    affine = np.asarray(affine, dtype=np.float64)
    if mask.ndim != 3 or min(mask.shape) < 2:
        raise ValueError("geometry evaluation needs a 3D grid with each dimension >= 2")
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or np.linalg.det(affine[:3, :3]) == 0:
        raise ValueError("affine must be finite, invertible, and 4x4")
    vertices, faces = volume_to_mesh(mask.astype(np.float32), affine)
    result = mesh_geometry(vertices, faces)
    result["foreground_voxels"] = int(mask.sum())
    result["foreground_touches_grid_boundary"] = bool(
        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
        or mask[:, :, 0].any() or mask[:, :, -1].any())
    return result


def geometry_comparison(prediction, target, affine):
    """Signed delta K = predicted minus target integral, valid for closed manifolds."""
    if np.shape(prediction) != np.shape(target):
        raise ValueError("prediction and target must have identical shapes")
    predicted = mask_geometry(prediction, affine)
    reference = mask_geometry(target, affine)
    valid = all(item["status"] == "closed_surface" for item in (predicted, reference))
    delta = (predicted["integrated_gaussian_curvature"] - reference["integrated_gaussian_curvature"]
             if valid else None)
    return {"definition": "angle_defect_on_unpadded_thresholded_mask_mesh",
            "prediction": predicted, "target": reference,
            "delta_integrated_gaussian_curvature": delta,
            "absolute_delta_integrated_gaussian_curvature": abs(delta) if valid else None,
            "delta_k_status": "valid" if valid else "requires_two_closed_manifold_surfaces"}
