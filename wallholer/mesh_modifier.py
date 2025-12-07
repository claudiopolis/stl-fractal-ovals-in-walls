"""
Mesh modification - applying oval holes to walls using boolean operations.
"""

import numpy as np
import trimesh
from typing import List, Dict
from scipy.spatial.transform import Rotation

from wallholer.wall_detector import Wall
from wallholer.hole_generator import Oval
from wallholer.config import Config


def apply_holes_to_walls(
    mesh: trimesh.Trimesh,
    walls: List[Wall],
    hole_patterns: Dict[int, List[Oval]],
    config: Config
) -> trimesh.Trimesh:
    """
    Apply oval holes to the specified walls using boolean operations.

    Args:
        mesh: Original mesh
        walls: List of all detected walls
        hole_patterns: Dictionary mapping wall_id to list of Ovals
        config: Configuration

    Returns:
        Modified mesh with holes
    """
    result_mesh = mesh.copy()

    # Process each wall
    for wall_id, ovals in hole_patterns.items():
        print(f"  Processing wall {wall_id} with {len(ovals)} ovals...")

        wall = walls[wall_id]

        # Create a union of all oval cutters for this wall
        cutters = []
        invalid_count = 0
        for i, oval in enumerate(ovals):
            cutter = create_oval_cylinder(oval, wall, config)
            if cutter is not None:
                # Check if it's a valid volume
                if not cutter.is_volume:
                    invalid_count += 1
                    if invalid_count == 1:  # Only print first warning
                        print(f"    WARNING: Cylinder {i} is not a valid volume!")
                        print(f"      watertight={cutter.is_watertight}, volume={cutter.is_volume}")
                else:
                    cutters.append(cutter)

            # Progress indicator for large numbers of ovals
            if (i + 1) % 10 == 0:
                print(f"    Created {i + 1}/{len(ovals)} cutters...")

        if invalid_count > 0:
            print(f"    Rejected {invalid_count} invalid cylinders")

        if not cutters:
            print(f"    Warning: No valid cutters created for wall {wall_id}")
            continue

        print(f"    Using {len(cutters)} valid cylinders")

        # Combine all cutters into one mesh
        print(f"    Combining {len(cutters)} cutters...")
        combined_cutter = trimesh.util.concatenate(cutters)

        # Perform boolean difference
        print(f"    Performing boolean subtraction...")
        print(f"    Input mesh before: {len(result_mesh.vertices)} vertices, {len(result_mesh.faces)} faces, volume={result_mesh.volume:.2f}")
        try:
            result_mesh = result_mesh.difference(combined_cutter, engine='manifold')
            print(f"    ✓ Boolean operation successful!")
            print(f"    Result: {len(result_mesh.vertices)} vertices, {len(result_mesh.faces)} faces")
            if hasattr(result_mesh, 'volume'):
                print(f"    Result volume: {result_mesh.volume:.2f} mm³")

            # Check if result is empty or invalid
            if len(result_mesh.vertices) == 0 or len(result_mesh.faces) == 0:
                print(f"    WARNING: Boolean operation produced empty mesh!")
                print(f"    This likely means the holes removed all material from the wall.")
                print(f"    Try: fewer holes, larger spacing, or smaller hole sizes.")
        except Exception as e:
            print(f"    Warning: Boolean operation failed with manifold engine: {e}")
            print(f"    Trying with blender engine...")
            try:
                result_mesh = result_mesh.difference(combined_cutter, engine='blender')
                print(f"    ✓ Boolean operation successful (blender)!")
                print(f"    Result: {len(result_mesh.vertices)} vertices, {len(result_mesh.faces)} faces")
            except Exception as e2:
                print(f"    Error: Boolean operation failed: {e2}")
                print(f"    Skipping wall {wall_id}")
                continue

    print(f"\nFinal mesh: {len(result_mesh.vertices)} vertices, {len(result_mesh.faces)} faces")
    return result_mesh


def create_oval_cylinder(oval: Oval, wall: Wall, config: Config) -> trimesh.Trimesh:
    """
    Create a 3D cylinder with oval cross-section to cut from the mesh.

    The cylinder extends through the wall thickness plus some margin.

    Args:
        oval: The oval hole specification
        wall: The wall this oval belongs to
        config: Configuration

    Returns:
        A trimesh object representing the oval cylinder
    """
    # Create an elliptical cylinder using a custom mesh
    # The cylinder's height is the wall thickness plus margin
    cylinder_height = wall.thickness + 4.0  # Add 4mm margin to ensure it cuts through

    # Create vertices for an elliptical cylinder
    radius_x = oval.width / 2
    radius_z = oval.length_z / 2
    sections = 32  # Number of facets around circumference

    # Generate ellipse points
    theta = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    x = radius_x * np.cos(theta)
    z = radius_z * np.sin(theta)

    # Create top and bottom circles
    top_y = cylinder_height / 2
    bottom_y = -cylinder_height / 2

    # Build vertices
    vertices = []
    # Bottom circle
    for i in range(sections):
        vertices.append([x[i], bottom_y, z[i]])
    # Top circle
    for i in range(sections):
        vertices.append([x[i], top_y, z[i]])
    # Center points for caps
    vertices.append([0, bottom_y, 0])  # bottom center
    vertices.append([0, top_y, 0])      # top center

    vertices = np.array(vertices)

    # Build faces
    faces = []
    # Side faces
    for i in range(sections):
        next_i = (i + 1) % sections
        # Two triangles per quad
        faces.append([i, next_i, sections + i])
        faces.append([next_i, sections + next_i, sections + i])

    # Bottom cap (pointing down)
    bottom_center = len(vertices) - 2
    for i in range(sections):
        next_i = (i + 1) % sections
        faces.append([bottom_center, next_i, i])

    # Top cap (pointing up)
    top_center = len(vertices) - 1
    for i in range(sections):
        next_i = (i + 1) % sections
        faces.append([top_center, sections + i, sections + next_i])

    faces = np.array(faces)

    # Create mesh
    cylinder = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Ensure it's watertight
    trimesh.repair.fill_holes(cylinder)
    try:
        trimesh.repair.fix_normals(cylinder)
    except:
        pass  # Fix normals might fail, continue anyway

    # Verify it's a valid volume
    if not cylinder.is_volume:
        print(f"      Warning: Cylinder is not a valid volume! Attempting to fix...")
        # Try to make it convex hull as fallback
        try:
            cylinder = cylinder.convex_hull
        except:
            return None

    # Now transform to correct position and orientation
    # Rotate to align with wall normal
    rotation = align_with_normal(wall.normal)
    cylinder.apply_transform(rotation)

    # Additional rotation to align oval's long axis with Z
    z_rotation = align_oval_with_z(wall.normal)
    cylinder.apply_transform(z_rotation)

    # Translate to position (oval.center is already in global coordinates)
    translation = trimesh.transformations.translation_matrix(oval.center)
    cylinder.apply_transform(translation)

    return cylinder


def align_with_normal(normal: np.ndarray) -> np.ndarray:
    """
    Create a transformation matrix to align the Y-axis with the given normal.

    Args:
        normal: The target normal vector

    Returns:
        4x4 transformation matrix
    """
    # Normalize the normal
    normal = normal / np.linalg.norm(normal)

    # Default direction is Y-axis
    y_axis = np.array([0, 1, 0])

    # If normal is already aligned with Y, return identity
    if np.allclose(normal, y_axis):
        return np.eye(4)
    if np.allclose(normal, -y_axis):
        # 180 degree rotation around X
        return trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])

    # Calculate rotation axis and angle
    rotation_axis = np.cross(y_axis, normal)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    rotation_angle = np.arccos(np.clip(np.dot(y_axis, normal), -1.0, 1.0))

    return trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)


def align_oval_with_z(wall_normal: np.ndarray) -> np.ndarray:
    """
    Create a transformation to align the oval's long axis with the Z direction.

    The oval is in the XZ plane of the cylinder, and we want the Z-axis of the oval
    to align with the global Z-axis.

    Args:
        wall_normal: The wall's normal vector

    Returns:
        4x4 transformation matrix
    """
    # Determine the rotation needed to align the oval properly
    # This is a rotation around the wall normal (Y-axis of the cylinder after first rotation)

    # Find the global Z projection onto the plane perpendicular to wall normal
    z_global = np.array([0, 0, 1])
    wall_normal = wall_normal / np.linalg.norm(wall_normal)

    # Project Z onto the wall plane
    z_projected = z_global - np.dot(z_global, wall_normal) * wall_normal

    if np.linalg.norm(z_projected) < 0.01:
        # Wall is horizontal, no additional rotation needed
        return np.eye(4)

    z_projected = z_projected / np.linalg.norm(z_projected)

    # We want the cylinder's Z-axis (which is in the XZ plane after scaling)
    # to align with z_projected
    # This is a rotation around the wall normal (Y-axis of cylinder)

    # The cylinder's local Z-axis is [0, 0, 1]
    # After the wall normal alignment, it's in the plane perpendicular to wall_normal
    # We need to rotate it around wall_normal to align with z_projected

    # Find the current Z direction (perpendicular to wall_normal, in XZ plane)
    # For simplicity, we'll compute the angle directly

    # Calculate angle in the wall plane
    # Using wall_normal as rotation axis
    angle = compute_rotation_angle_in_plane(wall_normal, z_projected)

    return trimesh.transformations.rotation_matrix(angle, wall_normal)


def compute_rotation_angle_in_plane(plane_normal: np.ndarray, target_direction: np.ndarray) -> float:
    """
    Compute the rotation angle around plane_normal to align with target_direction.

    Args:
        plane_normal: Normal to the plane of rotation
        target_direction: Direction to align with

    Returns:
        Rotation angle in radians
    """
    # This is a simplified version - for production you'd want more robust handling
    # For now, we'll use a simple heuristic based on the wall orientation

    # If wall is vertical (normal in XY plane), rotate to align with Z
    if abs(plane_normal[2]) < 0.1:
        # Wall is mostly vertical
        return 0.0  # Ovals already aligned with Z

    # If wall is horizontal, we need different logic
    return 0.0  # Default: no rotation
