"""
Wall detection algorithm for identifying thin flat walls in 3D meshes.
"""

import numpy as np
import trimesh
from typing import List, Dict, Any
from dataclasses import dataclass

from wallholer.config import Config


def cluster_faces_by_proximity(mesh: trimesh.Trimesh, face_indices: List[int], normal: np.ndarray) -> List[List[int]]:
    """
    Cluster faces by spatial proximity along the normal direction.

    This separates opposite walls (e.g., faces at X=0 vs X=100).

    Args:
        mesh: The mesh
        face_indices: Indices of faces to cluster
        normal: The normal direction these faces share

    Returns:
        List of clusters, where each cluster is a list of face indices
    """
    if len(face_indices) == 0:
        return []

    # Get face centroids
    face_centroids = mesh.triangles_center[face_indices]

    # Project centroids onto the normal axis
    positions = np.dot(face_centroids, normal)

    # Find distinct positions (cluster by proximity)
    # Use a simple threshold: if positions differ by more than 1% of model size, they're separate walls
    model_size = np.ptp(mesh.vertices, axis=0).max()
    threshold = model_size * 0.05  # 5% of model size

    # Sort positions
    sorted_indices = np.argsort(positions)
    sorted_positions = positions[sorted_indices]

    # Group faces that are close together
    clusters = []
    current_cluster = [face_indices[sorted_indices[0]]]
    current_position = sorted_positions[0]

    for i in range(1, len(sorted_indices)):
        pos = sorted_positions[i]
        if abs(pos - current_position) < threshold:
            # Same cluster
            current_cluster.append(face_indices[sorted_indices[i]])
        else:
            # New cluster
            clusters.append(current_cluster)
            current_cluster = [face_indices[sorted_indices[i]]]
            current_position = pos

    # Don't forget the last cluster
    clusters.append(current_cluster)

    return clusters


@dataclass
class Wall:
    """Represents a detected wall in the mesh."""
    id: int
    faces: np.ndarray  # Indices of faces that make up this wall
    vertices: np.ndarray  # Vertices of the wall
    normal: np.ndarray  # Average normal vector of the wall
    thickness: float  # Estimated thickness in mm
    bounds: np.ndarray  # Bounding box [min, max] for each axis
    center: np.ndarray  # Center point of the wall
    dimensions: np.ndarray  # Dimensions [x, y, z] of the wall


def detect_walls(mesh: trimesh.Trimesh, config: Config) -> List[Wall]:
    """
    Detect thin flat walls in the mesh.

    Strategy:
    1. Group faces by similar normals (to find flat regions)
    2. For each group, check if it forms a thin wall based on:
       - Thickness (one dimension much smaller than others)
       - Aspect ratio (flat, not cube-like)
    3. Return list of detected walls

    Args:
        mesh: The 3D mesh to analyze
        config: Configuration with detection thresholds

    Returns:
        List of detected Wall objects
    """
    walls = []
    wall_id = 0

    # Get face normals
    face_normals = mesh.face_normals

    # Group faces by similar normals (±15 degrees tolerance)
    # We'll use a simple clustering approach
    normal_tolerance = np.cos(np.radians(15))

    # Track which faces have been assigned to walls
    assigned_faces = set()

    # Try to find walls aligned with major axes
    major_axes = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([-1, 0, 0]),
        np.array([0, -1, 0]),
        np.array([0, 0, -1])
    ]

    print(f"DEBUG: Total faces in mesh: {len(face_normals)}")
    print(f"DEBUG: Normal tolerance (cos 15°): {normal_tolerance:.3f}")

    for axis in major_axes:
        # Find faces aligned with this axis
        alignment = np.abs(np.dot(face_normals, axis))
        aligned_faces = np.where(alignment > normal_tolerance)[0]

        # Remove already assigned faces
        aligned_faces = [f for f in aligned_faces if f not in assigned_faces]

        print(f"DEBUG: Axis {axis}: {len(aligned_faces)} aligned faces")

        if len(aligned_faces) < 4:  # Skip if too few faces (need at least 2 triangles)
            print(f"DEBUG:   Skipped - too few faces (need 4+)")
            continue

        # Cluster aligned faces by spatial proximity
        # This separates opposite walls (e.g., X=0 and X=100)
        face_clusters = cluster_faces_by_proximity(mesh, aligned_faces, axis)
        print(f"DEBUG:   Clustered into {len(face_clusters)} spatial groups")

        for cluster_idx, cluster_faces in enumerate(face_clusters):
            if len(cluster_faces) < 4:
                print(f"DEBUG:   Cluster {cluster_idx}: only {len(cluster_faces)} faces, skipping")
                continue

            # Get vertices for these faces
            face_indices = mesh.faces[cluster_faces]
            unique_vertices = np.unique(face_indices.flatten())
            wall_vertices = mesh.vertices[unique_vertices]

            # Calculate bounding box
            bounds = np.array([wall_vertices.min(axis=0), wall_vertices.max(axis=0)])
            dimensions = bounds[1] - bounds[0]

            # Check if this is a thin wall
            # One dimension should be significantly smaller than the other two
            sorted_dims = np.sort(dimensions)
            thickness = sorted_dims[0]
            width = sorted_dims[1]
            length = sorted_dims[2]

            print(f"DEBUG:   Cluster {cluster_idx} dimensions (sorted): {thickness:.2f} x {width:.2f} x {length:.2f} mm")
            print(f"DEBUG:   Checking: thickness <= {config.wall_thickness_threshold:.2f}, width >= {thickness * config.wall_aspect_ratio:.2f}, length >= {thickness * config.wall_aspect_ratio:.2f}")

            # Check minimum thickness (reject artifacts with 0 or near-0 thickness)
            if thickness < 0.5:  # Minimum 0.5mm thickness
                print(f"DEBUG:   Skipped - thickness {thickness:.2f} too small (min 0.5mm)")
                continue

            # Check thickness threshold
            if thickness > config.wall_thickness_threshold:
                print(f"DEBUG:   Skipped - thickness {thickness:.2f} > threshold {config.wall_thickness_threshold:.2f}")
                continue

            # Check aspect ratio
            if width < thickness * config.wall_aspect_ratio:
                print(f"DEBUG:   Skipped - width {width:.2f} < {thickness * config.wall_aspect_ratio:.2f}")
                continue
            if length < thickness * config.wall_aspect_ratio:
                print(f"DEBUG:   Skipped - length {length:.2f} < {thickness * config.wall_aspect_ratio:.2f}")
                continue

            # This looks like a wall!
            print(f"DEBUG:   ✓ WALL DETECTED! (ID: {wall_id})")
            wall = Wall(
                id=wall_id,
                faces=np.array(cluster_faces),
                vertices=wall_vertices,
                normal=axis,
                thickness=thickness,
                bounds=bounds,
                center=(bounds[0] + bounds[1]) / 2,
                dimensions=dimensions
            )
            walls.append(wall)

            # Mark faces as assigned
            assigned_faces.update(cluster_faces)

            wall_id += 1

    print(f"\nDEBUG: Total walls detected: {len(walls)}")
    return walls
