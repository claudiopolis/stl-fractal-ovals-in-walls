"""
Oval hole pattern generation algorithm.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from scipy.spatial import ConvexHull

from wallholer.wall_detector import Wall
from wallholer.config import Config


@dataclass
class Oval:
    """Represents an oval hole."""
    center: np.ndarray  # 3D center position in global coordinates
    center_2d: np.ndarray  # 2D position in wall-local coordinates [x, y]
    length_z: float  # Length along z-axis (local y-direction)
    width: float  # Width perpendicular to z (local x-direction)
    rotation: np.ndarray  # Rotation to align with wall and z-axis


def generate_hole_pattern(wall: Wall, config: Config, bottom_wall_id: int, mesh=None) -> List[Oval]:
    """
    Generate a fractal-like pattern of oval holes for a wall.

    Strategy:
    1. Determine the z-axis direction (perpendicular to bottom wall)
    2. Calculate oval sizes based on config (multiple sizes for fractal effect)
    3. Recursively place ovals:
       - Start with largest ovals
       - Fill remaining spaces with smaller ovals
       - Respect minimum wall thickness constraints

    Args:
        wall: The wall to generate holes for
        config: Configuration with oval parameters
        bottom_wall_id: ID of the wall representing the build plate

    Returns:
        List of Oval objects representing the hole pattern
    """
    print(f"    Generating holes for wall {wall.id}...")
    ovals = []

    # Determine wall orientation and z-axis
    wall_2d_bounds, z_axis, local_x, local_y, wall_vertices_2d = get_wall_2d_space(wall, config, mesh)

    # Apply inter-wall clearance: clip 2D bounds if 3D bounds were restricted
    # Check if wall.bounds is more restrictive than actual vertices
    vertices_bounds_3d = np.array([
        wall.vertices.min(axis=0),
        wall.vertices.max(axis=0)
    ])

    bounds_clipped = (
        np.any(wall.bounds[0] > vertices_bounds_3d[0] + 0.01) or
        np.any(wall.bounds[1] < vertices_bounds_3d[1] - 0.01)
    )

    if bounds_clipped:
        # Map 3D bounds clip to 2D bounds
        # local_y corresponds to Z-axis projected onto wall, so Y in 2D ~ Z in 3D
        # We need to clip the 2D bounds based on which 3D axes were clipped

        wall_2d_bounds_clipped = wall_2d_bounds.copy()

        # For each 3D axis (X, Y, Z), check if it was clipped
        for axis_3d in range(3):
            if (wall.bounds[0][axis_3d] > vertices_bounds_3d[0][axis_3d] + 0.01 or
                wall.bounds[1][axis_3d] < vertices_bounds_3d[1][axis_3d] - 0.01):

                # This 3D axis was clipped
                # Map clipped 3D coordinates to 2D

                # Create points at the clipped bounds for this axis
                # Use wall center for other axes
                point_min_3d = wall.center.copy()
                point_min_3d[axis_3d] = wall.bounds[0][axis_3d]

                point_max_3d = wall.center.copy()
                point_max_3d[axis_3d] = wall.bounds[1][axis_3d]

                # Project to 2D
                point_min_rel = point_min_3d - wall.center
                point_max_rel = point_max_3d - wall.center

                point_min_2d_x = np.dot(point_min_rel, local_x)
                point_min_2d_y = np.dot(point_min_rel, local_y)
                point_max_2d_x = np.dot(point_max_rel, local_x)
                point_max_2d_y = np.dot(point_max_rel, local_y)

                # Determine which 2D axis is affected (X or Y)
                # Check which 2D coordinate changed more
                delta_2d_x = abs(point_max_2d_x - point_min_2d_x)
                delta_2d_y = abs(point_max_2d_y - point_min_2d_y)

                if delta_2d_y > delta_2d_x:
                    # Y in 2D corresponds to this 3D axis
                    wall_2d_bounds_clipped[0][1] = max(wall_2d_bounds_clipped[0][1], min(point_min_2d_y, point_max_2d_y))
                    wall_2d_bounds_clipped[1][1] = min(wall_2d_bounds_clipped[1][1], max(point_min_2d_y, point_max_2d_y))
                elif delta_2d_x > 0.001:
                    # X in 2D corresponds to this 3D axis
                    wall_2d_bounds_clipped[0][0] = max(wall_2d_bounds_clipped[0][0], min(point_min_2d_x, point_max_2d_x))
                    wall_2d_bounds_clipped[1][0] = min(wall_2d_bounds_clipped[1][0], max(point_min_2d_x, point_max_2d_x))

        print(f"      Applied inter-wall clearance:")
        print(f"        Original 2D bounds: {wall_2d_bounds.tolist()}")
        print(f"        Clipped 2D bounds:  {wall_2d_bounds_clipped.tolist()}")

        wall_2d_bounds = wall_2d_bounds_clipped
        # DON'T clip wall_vertices_2d - keep the original diagonal edges
        # The rectangular bounds check (which happens first) will prevent ovals
        # from going outside the clipped bounds, and the polygon check will
        # prevent them from being bisected by diagonal edges

    # Calculate oval sizes (after determining final wall bounds)
    oval_sizes = calculate_oval_sizes(config, wall_2d_bounds)
    print(f"    Oval sizes: {[f'{l:.1f}x{w:.1f}mm' for l, w in oval_sizes]}")

    # Generate the pattern recursively
    min_spacing = config.min_wall_width

    # Track ovals by level for diagnostic output
    ovals_by_level = []

    # Start with the largest ovals
    for size_idx, (oval_length, oval_width) in enumerate(oval_sizes):
        # Generate and place ovals (added incrementally to ovals list)
        new_ovals = generate_oval_positions(
            wall,
            wall_2d_bounds,
            oval_length,
            oval_width,
            min_spacing,
            z_axis,
            local_x,
            local_y,
            ovals,  # Existing ovals - new ovals added to this list incrementally
            oval_sizes,  # All oval sizes for fractal spacing calculation
            wall_vertices_2d  # Wall polygon for boundary checking
        )

        print(f"      Size {size_idx} ({oval_length:.1f}x{oval_width:.1f}mm): {len(new_ovals)} positions")

        # Save level data for diagnostics
        ovals_by_level.append({
            'level': size_idx,
            'oval_length': oval_length,
            'oval_width': oval_width,
            'positions': [oval.center_2d.tolist() for oval in new_ovals]
        })

    # Save diagnostic data
    _save_diagnostic_data(wall.id, wall_2d_bounds, wall_vertices_2d, ovals_by_level)

    print(f"    Total ovals for wall {wall.id}: {len(ovals)}")
    return ovals


def _optimize_level0_size(wall_2d_bounds: np.ndarray, max_oval_length_z: float,
                          aspect_ratio: float, min_spacing: float) -> Tuple[float, float, int]:
    """
    Calculate optimal Level 0 oval size for perfect row packing using wavelength approach.

    Algorithm (Wavelength approach):
    1. n_x = floor(W_wall / (max_oval_width + min_spacing))
    2. wavelength = (W_wall - min_spacing) / n_x  (center-to-center spacing)
    3. oval_width = wavelength - min_spacing
    4. oval_length = oval_width * aspect_ratio
    5. If oval_length > max, constrain to max and recalculate oval_width

    This ensures equal edge clearances on left/right automatically.

    Args:
        wall_2d_bounds: Wall bounds [[min_x, min_y], [max_x, max_y]]
        max_oval_length_z: Maximum allowed oval length
        aspect_ratio: Oval aspect ratio (length / width)
        min_spacing: Minimum wall width

    Returns:
        (length, width, n_x) tuple for optimal Level 0 size and count
    """
    wall_min_x, wall_min_y = wall_2d_bounds[0]
    wall_max_x, wall_max_y = wall_2d_bounds[1]
    W_wall = wall_max_x - wall_min_x

    # Calculate max oval width from max length
    max_oval_width = max_oval_length_z / aspect_ratio

    # Calculate number of ovals using wavelength approach
    # To get exactly min_spacing edge clearances, use:
    # n_x = ceil((W_wall - min_spacing) / (max_oval_width + min_spacing))
    # This gives the smallest n_x where oval_length ≤ max, maximizing oval size
    import math
    n_x = math.ceil((W_wall - min_spacing) / (max_oval_width + min_spacing))
    if n_x < 1:
        n_x = 1

    # Use wavelength approach: wavelength = (W_wall - min_spacing) / n_x
    wavelength = (W_wall - min_spacing) / n_x
    W = wavelength - min_spacing
    L = W * aspect_ratio

    # Calculate actual edge clearance with wavelength approach
    total_occupied = n_x * W + (n_x - 1) * min_spacing
    edge_clearance = (W_wall - total_occupied) / 2

    print(f"      Optimized Level 0: {L:.2f} x {W:.2f} mm ({n_x} ovals per row)")
    print(f"      Edge clearances: {edge_clearance:.2f}mm on left/right")

    return L, W, n_x


def calculate_oval_sizes(config: Config, wall_2d_bounds: np.ndarray = None) -> List[Tuple[float, float]]:
    """
    Calculate the different oval sizes to use.

    Algorithm:
    1. Optimize Level 0 size for perfect row packing (≤ max_oval_length_z)
    2. Level 1: fits in horizontal/vertical gaps between Level 0 ovals
    3. Level 2+: fits diagonally between Level 0 ovals, accounting for Level 1
    4. Continue until num_oval_sizes reached or no smaller oval can fit

    Args:
        config: Configuration with oval parameters
        wall_2d_bounds: Optional wall bounds [[min_x, min_y], [max_x, max_y]] for optimization

    Returns list of (length, width) tuples, sorted from largest to smallest.
    """
    sizes = []
    min_spacing = config.min_wall_width
    aspect_ratio = config.oval_aspect_ratio

    # Calculate optimal Level 0 size for perfect row packing
    if wall_2d_bounds is not None:
        level0_length, level0_width, _ = _optimize_level0_size(
            wall_2d_bounds, config.max_oval_length_z, aspect_ratio, min_spacing
        )
    else:
        # Fallback: use max directly if no bounds provided
        level0_length = config.max_oval_length_z
        level0_width = level0_length / aspect_ratio

    sizes.append((level0_length, level0_width))

    # Calculate Level 0 grid spacing
    level0_spacing_x = level0_width + min_spacing
    level0_spacing_y = level0_length + min_spacing

    # Calculate subsequent sizes based on Level 0
    # Each smaller level is calculated to fit in the diamond-shaped space at the
    # center of a 2x2 grid of the previous level's ovals
    # Uses binary search to find the maximum size that maintains min_spacing clearance
    for i in range(1, config.num_oval_sizes):
        if i == 1:
            # Level 1: fits in diamond space at center of 2x2 grid of Level 0 ovals
            corner_width = level0_width
            corner_length = level0_length
            spacing_x = level0_spacing_x
            spacing_y = level0_spacing_y
        else:
            # Level 2+: fits in diamond space at center of 2x2 grid of previous level
            prev_length, prev_width = sizes[i-1]
            corner_width = prev_width
            corner_length = prev_length
            spacing_x = prev_width + min_spacing
            spacing_y = prev_length + min_spacing

        # Binary search for maximum oval size that maintains min_spacing clearance
        # Set up 4 corner oval positions in a 2x2 grid
        center_pos_x = spacing_x / 2
        center_pos_y = spacing_y / 2

        # Binary search for max length (width follows from aspect ratio)
        min_length = 0.5  # Start very small
        max_length = min(spacing_x, spacing_y)  # Can't be bigger than spacing

        for _ in range(50):  # 50 iterations for precision
            trial_length = (min_length + max_length) / 2
            trial_width = trial_length / aspect_ratio

            # Check if this size maintains clearance to all 4 corner ovals
            # Corner ovals are at (0,0), (spacing_x,0), (0,spacing_y), (spacing_x,spacing_y)
            violates = False
            for corner_x in [0, spacing_x]:
                for corner_y in [0, spacing_y]:
                    # Calculate distance from center to this corner
                    dx = abs(center_pos_x - corner_x)
                    dy = abs(center_pos_y - corner_y)

                    # Use elliptical distance check (same as check_overlap function)
                    combined_x = (trial_width + corner_width) / 2 + min_spacing
                    combined_y = (trial_length + corner_length) / 2 + min_spacing
                    normalized_dist_sq = (dx / combined_x) ** 2 + (dy / combined_y) ** 2

                    # Target normalized_dist_sq >= 1.005 to leave margin for floating point errors
                    # (matching 0.998 threshold would create ovals at exact boundary)
                    if normalized_dist_sq < 1.005:
                        violates = True
                        break
                if violates:
                    break

            if violates:
                max_length = trial_length  # Too big, reduce
            else:
                min_length = trial_length  # OK, try bigger

        max_next_length = min_length
        max_next_width = max_next_length / aspect_ratio

        # Check if a valid oval can fit
        if max_next_width <= 0 or max_next_length <= 0:
            print(f"    Cannot fit oval size {i+1}: gap too small")
            break

        # The next oval should maintain the aspect ratio
        # We need: next_length / next_width = aspect_ratio
        # We have constraints: next_width <= max_next_width, next_length <= max_next_length

        # Try fitting with width constraint:
        next_width_from_width = max_next_width
        next_length_from_width = next_width_from_width * aspect_ratio

        # Try fitting with length constraint:
        next_length_from_length = max_next_length
        next_width_from_length = next_length_from_length / aspect_ratio

        # Use whichever is smaller (limiting constraint)
        if next_length_from_width <= max_next_length:
            # Width is the limiting constraint
            next_width = next_width_from_width
            next_length = next_length_from_width
        else:
            # Length is the limiting constraint
            next_width = next_width_from_length
            next_length = next_length_from_length

        # Verify it's large enough to be useful (at least 1mm)
        if next_width < 1.0 or next_length < 1.0:
            print(f"    Oval size {i+1} too small: {next_length:.2f}x{next_width:.2f}mm")
            break

        sizes.append((next_length, next_width))

    return sizes


def _extract_boundary_edges_2d(wall, mesh, local_x, local_y):
    """
    Extract boundary edges from wall mesh and project to 2D.

    Boundary edges are edges that belong to only one triangle face.
    This gives the TRUE wall boundary including diagonal cuts.

    Args:
        wall: Wall object with face indices
        mesh: Trimesh mesh object
        local_x, local_y: Local coordinate system for projecting to 2D

    Returns:
        Ordered 2D polygon vertices representing the wall boundary
    """
    if mesh is None:
        return None

    # Get wall faces from mesh
    wall_faces = mesh.faces[wall.faces]

    # Build edge-to-face mapping
    from collections import defaultdict
    edge_to_faces = defaultdict(list)

    for face_idx, face in enumerate(wall_faces):
        # Each triangle has 3 edges
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]
        for edge in edges:
            edge_to_faces[edge].append(face_idx)

    # Find boundary edges (edges with only one adjacent face)
    boundary_edges = []
    for edge, faces in edge_to_faces.items():
        if len(faces) == 1:
            boundary_edges.append(edge)

    if len(boundary_edges) == 0:
        return None

    # Order boundary edges to form a closed loop
    # Start with first edge
    ordered_vertices = [boundary_edges[0][0], boundary_edges[0][1]]
    used_edges = {0}

    # Build adjacency map for quick lookup
    vertex_to_edges = defaultdict(list)
    for i, edge in enumerate(boundary_edges):
        vertex_to_edges[edge[0]].append(i)
        vertex_to_edges[edge[1]].append(i)

    # Follow the loop
    while len(used_edges) < len(boundary_edges):
        current_vertex = ordered_vertices[-1]

        # Find next edge connected to current vertex
        found = False
        for edge_idx in vertex_to_edges[current_vertex]:
            if edge_idx not in used_edges:
                edge = boundary_edges[edge_idx]
                # Add the other vertex of this edge
                next_vertex = edge[1] if edge[0] == current_vertex else edge[0]
                ordered_vertices.append(next_vertex)
                used_edges.add(edge_idx)
                found = True
                break

        if not found:
            break  # Reached end or disconnected boundary

    # Get 3D coordinates of boundary vertices
    boundary_vertices_3d = mesh.vertices[ordered_vertices]

    # Project to 2D in wall-local coordinates
    vertices_relative = boundary_vertices_3d - wall.center
    boundary_vertices_2d = np.column_stack([
        np.dot(vertices_relative, local_x),
        np.dot(vertices_relative, local_y)
    ])

    return boundary_vertices_2d


def get_wall_2d_space(wall: Wall, config: Config, mesh=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the 2D parametric space for the wall and determine z-axis.

    Returns:
        - 2D bounds [[min_x, min_y], [max_x, max_y]]
        - z_axis direction (global up, perpendicular to build plate)
        - local_x direction (in wall plane)
        - local_y direction (in wall plane)
        - wall_vertices_2d: Wall boundary polygon in 2D local coordinates (N x 2 array)
    """
    # Assume z-axis is global Z (perpendicular to XY build plate)
    z_axis = np.array([0, 0, 1])

    # Get wall normal
    wall_normal = wall.normal

    # Create local coordinate system in the wall
    # local_y aligns with z_axis projected onto wall
    # local_x is perpendicular to both

    # Project z_axis onto wall plane
    z_projected = z_axis - np.dot(z_axis, wall_normal) * wall_normal
    if np.linalg.norm(z_projected) < 0.01:
        # Wall is horizontal, use different approach
        local_y = np.array([0, 1, 0])
        local_y = local_y - np.dot(local_y, wall_normal) * wall_normal
    else:
        local_y = z_projected

    local_y = local_y / np.linalg.norm(local_y)
    local_x = np.cross(wall_normal, local_y)
    local_x = local_x / np.linalg.norm(local_x)

    # Extract true boundary polygon from mesh
    wall_polygon_2d = _extract_boundary_edges_2d(wall, mesh, local_x, local_y)

    if wall_polygon_2d is None:
        # Fallback: use all vertices (old behavior)
        vertices_relative = wall.vertices - wall.center
        vertices_2d = np.column_stack([
            np.dot(vertices_relative, local_x),
            np.dot(vertices_relative, local_y)
        ])

        # Order by angle
        centroid_2d = vertices_2d.mean(axis=0)
        angles = np.arctan2(vertices_2d[:, 1] - centroid_2d[1],
                            vertices_2d[:, 0] - centroid_2d[0])
        sorted_indices = np.argsort(angles)
        wall_polygon_2d = vertices_2d[sorted_indices]

    # Calculate bounds from polygon
    bounds_2d = np.array([
        wall_polygon_2d.min(axis=0),
        wall_polygon_2d.max(axis=0)
    ])

    return bounds_2d, z_axis, local_x, local_y, wall_polygon_2d


def build_occupancy_grid(bounds_2d: np.ndarray, existing_ovals: List[Oval], min_spacing: float,
                         grid_resolution: float = 1.0) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Build a 2D occupancy grid showing which areas are occupied by existing ovals.

    Args:
        bounds_2d: 2D bounds [[min_x, min_y], [max_x, max_y]]
        existing_ovals: List of placed ovals with center_2d coordinates
        min_spacing: Minimum spacing to maintain around ovals
        grid_resolution: Size of each grid cell in mm

    Returns:
        Tuple of (occupancy_grid, grid_min_x, grid_min_y, grid_shape)
        occupancy_grid: 2D boolean array (True = occupied)
        grid_min_x, grid_min_y: Origin of the grid
        grid_shape: (nx, ny) number of cells in each direction
    """
    # Determine grid bounds and size
    grid_min_x, grid_min_y = bounds_2d[0]
    grid_max_x, grid_max_y = bounds_2d[1]

    nx = int(np.ceil((grid_max_x - grid_min_x) / grid_resolution)) + 1
    ny = int(np.ceil((grid_max_y - grid_min_y) / grid_resolution)) + 1

    occupancy_grid = np.zeros((ny, nx), dtype=bool)

    # Mark cells occupied by existing ovals (with buffer)
    for oval in existing_ovals:
        oval_x, oval_y = oval.center_2d

        # Calculate oval's bounding box with spacing buffer
        half_width = oval.width / 2 + min_spacing
        half_length = oval.length_z / 2 + min_spacing

        # Convert to grid indices
        ix_min = int(np.floor((oval_x - half_width - grid_min_x) / grid_resolution))
        ix_max = int(np.ceil((oval_x + half_width - grid_min_x) / grid_resolution))
        iy_min = int(np.floor((oval_y - half_length - grid_min_y) / grid_resolution))
        iy_max = int(np.ceil((oval_y + half_length - grid_min_y) / grid_resolution))

        # Clamp to grid bounds
        ix_min = max(0, ix_min)
        ix_max = min(nx, ix_max)
        iy_min = max(0, iy_min)
        iy_max = min(ny, iy_max)

        # Mark cells as occupied
        occupancy_grid[iy_min:iy_max, ix_min:ix_max] = True

    return occupancy_grid, grid_min_x, grid_min_y, (nx, ny)


def generate_oval_positions(
    wall: Wall,
    bounds_2d: np.ndarray,
    oval_length: float,
    oval_width: float,
    min_spacing: float,
    z_axis: np.ndarray,
    local_x: np.ndarray,
    local_y: np.ndarray,
    existing_ovals: List[Oval],
    oval_sizes: List[Tuple[float, float]] = None,
    wall_vertices_2d: np.ndarray = None
) -> List[Oval]:
    """
    Generate and place ovals of a given size within the wall bounds.

    Ovals are added incrementally to existing_ovals as they're validated,
    ensuring min_wall_width spacing is maintained between all ovals (same level and different levels).

    Algorithm:
    - Fractal grid with spacing = oval_size
    - Check each candidate position against ALL existing ovals (including same-level)
    - Add valid ovals immediately to existing_ovals list
    - This ensures proper spacing enforcement at all scales

    Returns:
        List of newly created Oval objects
    """
    new_ovals = []

    # Determine the level (0 = largest, 1 = next size down, etc.)
    level = 0
    if oval_sizes:
        # Find which level this oval size is
        for i, (length, width) in enumerate(oval_sizes):
            if abs(length - oval_length) < 0.01 and abs(width - oval_width) < 0.01:
                level = i
                break

    # Generate candidate positions
    candidate_positions = []

    # Calculate grid spacing for this level
    spacing_x = oval_width + min_spacing
    spacing_y = oval_length + min_spacing
    print(f"        Level {level}: tight-packing grid, spacing {spacing_x:.1f} x {spacing_y:.1f} mm")

    # For Level 0: Use optimized placement with perfect edge packing
    # For Level 1+: Use fractal placement
    if level == 0:
        # Calculate number of ovals per row for perfect packing
        W_wall = bounds_2d[1][0] - bounds_2d[0][0]
        H_wall = bounds_2d[1][1] - bounds_2d[0][1]

        # Calculate n_x using wavelength approach to ensure exactly min_spacing edge clearances
        import math
        n_x = math.ceil((W_wall - min_spacing) / (oval_width + min_spacing))
        if n_x < 1:
            n_x = 1

        # Use wavelength approach for perfect spacing with equal edge clearances
        # wavelength = center-to-center spacing
        wavelength = (W_wall - min_spacing) / n_x

        # Calculate edge clearance (automatically equal on left/right)
        edge_clearance = (W_wall - n_x * wavelength + min_spacing) / 2

        # Place ovals using wavelength spacing
        x_positions = []
        for i in range(n_x):
            x = bounds_2d[0][0] + edge_clearance + oval_width/2 + i * wavelength
            x_positions.append(x)

        # For Y: place rows from bottom with min_spacing
        y_positions = []
        y = bounds_2d[0][1] + min_spacing + oval_length/2
        while y + oval_length/2 + min_spacing <= bounds_2d[1][1]:
            y_positions.append(y)
            y += oval_length + min_spacing

        print(f"        Level {level}: optimized grid, {n_x} x {len(y_positions)} ovals")

        for x_2d in x_positions:
            for y_2d in y_positions:
                candidate_positions.append(np.array([x_2d, y_2d]))
    else:
        # Level 1+: Use existing fractal logic
        # Calculate grid boundaries (stay clear of wall edges)
        x_min = bounds_2d[0][0] + oval_width/2 + min_spacing
        y_min = bounds_2d[0][1] + oval_length/2 + min_spacing
        x_max = bounds_2d[1][0] - oval_width/2 - min_spacing
        y_max = bounds_2d[1][1] - oval_length/2 - min_spacing

        # Explicitly add diagonal center positions from previous level
        diagonal_centers = _calculate_diagonal_centers(existing_ovals, level, bounds_2d, min_spacing)
        print(f"        Level {level}: adding {len(diagonal_centers)} fractal diagonal centers")
        candidate_positions.extend(diagonal_centers)

        # Add regular grid of candidate positions for space-filling
        if x_max > x_min and y_max > y_min:
            x_positions = np.arange(x_min, x_max + spacing_x/2, spacing_x)
            y_positions = np.arange(y_min, y_max + spacing_y/2, spacing_y)

            for x_2d in x_positions:
                for y_2d in y_positions:
                    candidate_positions.append(np.array([x_2d, y_2d]))

    # Try each candidate position
    for pos_2d in candidate_positions:
        # Check if oval overlaps with any existing ovals (including same-level)
        if check_overlap(pos_2d, oval_length, oval_width, existing_ovals, min_spacing):
            continue  # Overlaps, skip

        # Check if oval would be too close to wall boundaries
        if not check_wall_boundary_clearance(pos_2d[0], pos_2d[1], oval_length, oval_width,
                                            bounds_2d, min_spacing, wall_vertices_2d):
            continue  # Too close to edge, skip

        # Convert to 3D position in global coordinates
        pos_3d = wall.center + pos_2d[0] * local_x + pos_2d[1] * local_y

        # Create oval and add it immediately to both lists
        oval = Oval(
            center=pos_3d,
            center_2d=pos_2d,
            length_z=oval_length,
            width=oval_width,
            rotation=create_rotation_matrix(z_axis, wall.normal)
        )
        existing_ovals.append(oval)  # Add to existing list immediately
        new_ovals.append(oval)        # Track newly created ovals

    return new_ovals


def check_overlap(pos_2d: np.ndarray, length: float, width: float, existing_ovals: List[Oval], min_spacing: float) -> bool:
    """
    Check if a new oval at the given position would overlap with existing ovals.

    Uses elliptical distance calculation in 2D wall-local space, considering
    the actual dimensions of both ovals. Enforces consistent min_spacing clearance
    between ALL ovals regardless of size.

    Args:
        pos_2d: 2D position [x, y] in wall-local coordinates
        length: Length (along local_y/z-axis) of new oval
        width: Width (along local_x) of new oval
        existing_ovals: List of already placed ovals
        min_spacing: Minimum required spacing between all ovals

    Returns:
        True if there is overlap, False otherwise.
    """
    pos_2d_x, pos_2d_y = pos_2d

    for oval in existing_ovals:
        # Use stored 2D coordinates
        oval_2d_x, oval_2d_y = oval.center_2d

        # Calculate center-to-center distances along each axis
        dx = abs(pos_2d_x - oval_2d_x)
        dy = abs(pos_2d_y - oval_2d_y)

        # Use normalized elliptical distance metric
        # Each dimension is normalized by the sum of half-widths/lengths plus clearance
        # This properly models elliptical geometry instead of rectangular bounding boxes
        combined_x = (width + oval.width) / 2 + min_spacing
        combined_y = (length + oval.length_z) / 2 + min_spacing

        # Elliptical distance check: point (dx, dy) is inside exclusion ellipse if:
        # (dx/combined_x)² + (dy/combined_y)² < 1
        # Use 0.998 threshold to account for floating point precision errors
        normalized_dist_sq = (dx / combined_x) ** 2 + (dy / combined_y) ** 2

        if normalized_dist_sq < 0.998:
            return True  # Overlap detected

    return False


def point_to_segment_distance(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    """
    Calculate minimum distance from a point to a line segment.

    Args:
        point: 2D point [x, y]
        seg_start: Segment start point [x, y]
        seg_end: Segment end point [x, y]

    Returns:
        Minimum distance from point to segment
    """
    # Vector from start to end
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq < 1e-6:
        # Degenerate segment (point)
        return np.linalg.norm(point - seg_start)

    # Project point onto line, clamped to segment
    t = np.clip(np.dot(point - seg_start, seg_vec) / seg_len_sq, 0.0, 1.0)
    projection = seg_start + t * seg_vec

    return np.linalg.norm(point - projection)


def check_wall_boundary_clearance(pos_2d_x: float, pos_2d_y: float, length: float, width: float,
                                   wall_bounds_2d: np.ndarray, min_spacing: float,
                                   wall_vertices_2d: np.ndarray = None) -> bool:
    """
    Check if an oval has sufficient clearance from wall boundaries.

    If wall_vertices_2d is provided, checks against actual wall polygon edges.
    Otherwise, falls back to checking against rectangular bounding box.

    Args:
        pos_2d_x: X-coordinate in 2D wall-local space
        pos_2d_y: Y-coordinate in 2D wall-local space
        length: Length (along y-axis) of oval
        width: Width (along x-axis) of oval
        wall_bounds_2d: 2D bounding box [[min_x, min_y], [max_x, max_y]]
        min_spacing: Minimum required clearance from wall edges
        wall_vertices_2d: Optional Nx2 array of wall polygon vertices in 2D

    Returns:
        True if clearance is sufficient, False otherwise
    """
    pos_2d = np.array([pos_2d_x, pos_2d_y])

    # First, check against rectangular bounds (mandatory)
    # This ensures ovals don't extend beyond the clipped inter-wall clearance bounds
    oval_min_x = pos_2d_x - width / 2 - min_spacing
    oval_max_x = pos_2d_x + width / 2 + min_spacing
    oval_min_y = pos_2d_y - length / 2 - min_spacing
    oval_max_y = pos_2d_y + length / 2 + min_spacing

    wall_min_x, wall_min_y = wall_bounds_2d[0]
    wall_max_x, wall_max_y = wall_bounds_2d[1]

    if (oval_min_x < wall_min_x or oval_max_x > wall_max_x or
        oval_min_y < wall_min_y or oval_max_y > wall_max_y):
        return False  # Outside rectangular bounds

    # If we have actual wall polygon, check if oval is fully inside (not bisected by diagonal edges)
    if wall_vertices_2d is not None and len(wall_vertices_2d) > 2:
        # Point-in-polygon check: Every point on oval perimeter must be INSIDE the wall polygon
        # This prevents ovals from being bisected by diagonal edges
        import math
        from matplotlib.path import Path

        # Create path from wall polygon
        wall_path = Path(wall_vertices_2d)

        # Sample oval perimeter at 5° intervals (72 points) - sufficient to detect bisection
        # Also sample points OUTWARD by min_spacing to ensure clearance from wall boundary
        perimeter_points = []

        # 10% tolerance on min_spacing as requested
        tolerance = 0.9
        clearance = min_spacing * tolerance

        for angle_deg in range(0, 360, 5):  # 5° intervals = 72 points
            angle_rad = math.radians(angle_deg)

            # Oval edge point (at perimeter)
            px_edge = pos_2d_x + (width / 2) * math.cos(angle_rad)
            py_edge = pos_2d_y + (length / 2) * math.sin(angle_rad)
            perimeter_points.append([px_edge, py_edge])

            # Clearance point OUTWARD from oval edge by min_spacing
            # This ensures min_wall_width clearance between oval edge and wall boundary
            px_clearance = pos_2d_x + (width / 2 + clearance) * math.cos(angle_rad)
            py_clearance = pos_2d_y + (length / 2 + clearance) * math.sin(angle_rad)
            perimeter_points.append([px_clearance, py_clearance])

        # Check if all points are inside the wall polygon
        points_inside = wall_path.contains_points(perimeter_points)

        if not np.all(points_inside):
            return False  # Some perimeter points are outside wall polygon (oval would be bisected)

        return True  # All perimeter points inside wall polygon

    # No fallback - wall_vertices_2d should always be provided
    return False  # Reject if no polygon provided


def create_rotation_matrix(z_axis: np.ndarray, wall_normal: np.ndarray) -> np.ndarray:
    """
    Create a rotation matrix to align an oval with the z-axis and wall.

    The oval's long axis should align with z_axis,
    and it should lie in the plane perpendicular to wall_normal.
    """
    # For now, return identity - actual rotation will be handled during mesh modification
    return np.eye(3)


def _calculate_diagonal_centers(existing_ovals: List[Oval], current_level: int,
                                bounds_2d: np.ndarray, min_spacing: float) -> List[np.ndarray]:
    """
    Calculate diagonal center positions from previous level ovals for fractal pattern.

    Creates an extended grid that fills the wall bounds and places diagonal centers
    at ALL 2x2 grid cells where at least 1 corner has a previous level oval.
    This creates true fractal self-similarity.

    Args:
        existing_ovals: All ovals placed so far
        current_level: Current level being processed (1, 2, 3, ...)
        bounds_2d: Wall bounds [[min_x, min_y], [max_x, max_y]]
        min_spacing: Minimum spacing between ovals (min_wall_width)

    Returns:
        List of 2D position arrays representing diagonal centers
    """
    if current_level == 0 or len(existing_ovals) == 0:
        return []

    # Group ovals by their size to identify which level they belong to
    ovals_by_size = {}
    for oval in existing_ovals:
        size_key = (round(oval.length_z, 1), round(oval.width, 1))
        if size_key not in ovals_by_size:
            ovals_by_size[size_key] = []
        ovals_by_size[size_key].append(oval)

    # Sort by size (largest first)
    sorted_sizes = sorted(ovals_by_size.keys(), key=lambda k: k[0] * k[1], reverse=True)

    # Get ovals from previous level
    if current_level - 1 >= len(sorted_sizes):
        return []

    prev_level_size = sorted_sizes[current_level - 1]
    prev_level_ovals = ovals_by_size[prev_level_size]
    prev_level_length, prev_level_width = prev_level_size

    # Extract 2D positions
    positions = np.array([oval.center_2d for oval in prev_level_ovals])

    if len(positions) == 0:
        return []

    # Calculate grid spacing from previous level oval size
    spacing_x = prev_level_width + min_spacing
    spacing_y = prev_level_length + min_spacing

    # Find unique X and Y coordinates from actual oval positions
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    unique_x = sorted(np.unique(np.round(x_coords, 1)))
    unique_y = sorted(np.unique(np.round(y_coords, 1)))

    if len(unique_x) == 0 or len(unique_y) == 0:
        return []

    # Extend grid to fill wall boundaries
    wall_min_x, wall_min_y = bounds_2d[0]
    wall_max_x, wall_max_y = bounds_2d[1]

    # Start from outermost actual ovals and extend outward
    grid_x_min = unique_x[0]
    grid_x_max = unique_x[-1]
    grid_y_min = unique_y[0]
    grid_y_max = unique_y[-1]

    # Extend left - go one cell beyond wall boundary for edge diagonal centers
    extended_x = list(unique_x)
    x_left = grid_x_min - spacing_x
    while x_left >= wall_min_x:
        extended_x.insert(0, round(x_left, 1))
        x_left -= spacing_x
    # Add one more cell beyond boundary
    extended_x.insert(0, round(x_left, 1))

    # Extend right - go one cell beyond wall boundary for edge diagonal centers
    x_right = grid_x_max + spacing_x
    while x_right <= wall_max_x:
        extended_x.append(round(x_right, 1))
        x_right += spacing_x
    # Add one more cell beyond boundary
    extended_x.append(round(x_right, 1))

    # Extend down - go one cell beyond wall boundary for edge diagonal centers
    extended_y = list(unique_y)
    y_down = grid_y_min - spacing_y
    while y_down >= wall_min_y:
        extended_y.insert(0, round(y_down, 1))
        y_down -= spacing_y
    # Add one more cell beyond boundary
    extended_y.insert(0, round(y_down, 1))

    # Extend up - go one cell beyond wall boundary for edge diagonal centers
    y_up = grid_y_max + spacing_y
    while y_up <= wall_max_y:
        extended_y.append(round(y_up, 1))
        y_up += spacing_y
    # Add one more cell beyond boundary
    extended_y.append(round(y_up, 1))

    # For each potential 2x2 grid cell in extended grid
    diagonal_centers = []
    tolerance = 1.0  # mm - tolerance for finding ovals at grid positions

    for i in range(len(extended_x) - 1):
        for j in range(len(extended_y) - 1):
            x1, x2 = extended_x[i], extended_x[i + 1]
            y1, y2 = extended_y[j], extended_y[j + 1]

            # Check if at least 1 of the 4 corners has an oval
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            has_at_least_one_corner = False

            for cx, cy in corners:
                # Find if any actual oval position is close to this corner
                distances = np.sqrt((positions[:, 0] - cx)**2 + (positions[:, 1] - cy)**2)
                if np.min(distances) <= tolerance:
                    has_at_least_one_corner = True
                    break

            if has_at_least_one_corner:
                # Calculate diagonal center (midpoint of the 2x2 grid cell)
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                diagonal_centers.append(np.array([center_x, center_y]))

    return diagonal_centers


def _save_diagnostic_data(wall_id: int, bounds_2d: np.ndarray, wall_vertices_2d: np.ndarray, ovals_by_level: list):
    """
    Save diagnostic data for visualization and debugging.

    Args:
        wall_id: Wall identifier
        bounds_2d: 2D bounds [[min_x, min_y], [max_x, max_y]]
        ovals_by_level: List of dicts with level data
    """
    import json
    import os

    diagnostic_data = {
        'wall_id': wall_id,
        'bounds_2d': bounds_2d.tolist(),
        'levels': ovals_by_level
    }

    output_file = f"diagnostic_wall_{wall_id}.json"
    with open(output_file, 'w') as f:
        json.dump(diagnostic_data, f, indent=2)

    print(f"      Diagnostic data saved to {output_file}")

    # Generate PNG visualization
    _generate_diagnostic_png(wall_id, bounds_2d, wall_vertices_2d, ovals_by_level)


def _generate_diagnostic_png(wall_id: int, bounds_2d: np.ndarray, wall_vertices_2d: np.ndarray, ovals_by_level: list):
    """
    Generate PNG visualization of oval placement for debugging.

    Args:
        wall_id: Wall identifier
        bounds_2d: 2D bounds [[min_x, min_y], [max_x, max_y]]
        wall_vertices_2d: Actual wall polygon vertices
        ovals_by_level: List of dicts with level data
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw wall boundary
    wall_min_x, wall_min_y = bounds_2d[0]
    wall_max_x, wall_max_y = bounds_2d[1]
    wall_width = wall_max_x - wall_min_x
    wall_height = wall_max_y - wall_min_y

    # Draw rectangular bounds (inter-wall clearance bounds)
    wall_rect = patches.Rectangle((wall_min_x, wall_min_y), wall_width, wall_height,
                                   linewidth=1, edgecolor='gray', linestyle='--', facecolor='none', alpha=0.5,
                                   label='Inter-wall clearance bounds')
    ax.add_patch(wall_rect)

    # Draw actual wall polygon (with diagonal edges)
    if wall_vertices_2d is not None and len(wall_vertices_2d) > 2:
        wall_polygon = patches.Polygon(wall_vertices_2d, closed=True,
                                       linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.3,
                                       label='Actual wall boundary')
        ax.add_patch(wall_polygon)

    # Colors for different levels
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    # Draw ovals for each level with unique IDs
    oval_id = 0
    for level_data in ovals_by_level:
        level = level_data['level']
        oval_length = level_data['oval_length']
        oval_width = level_data['oval_width']
        positions = level_data['positions']
        color = colors[level % len(colors)]

        for pos in positions:
            x, y = pos
            # Draw ellipse (width is along X, length is along Y)
            ellipse = patches.Ellipse((x, y), oval_width, oval_length,
                                     linewidth=1, edgecolor=color, facecolor='none', alpha=0.7)
            ax.add_patch(ellipse)

            # Add oval ID text at center
            ax.text(x, y, str(oval_id),
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=8, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

            oval_id += 1

        # Add legend entry
        ax.plot([], [], color=color, linewidth=2,
               label=f'Level {level}: {oval_length:.1f}x{oval_width:.1f}mm ({len(positions)} ovals)')

    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(wall_min_x - 10, wall_max_x + 10)
    ax.set_ylim(wall_min_y - 10, wall_max_y + 10)

    # Labels and title
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title(f'Wall {wall_id} - Oval Placement Diagnostic\n'
                f'Wall dimensions: {wall_width:.1f} x {wall_height:.1f} mm', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Save PNG
    output_file = f"diagnostic_wall_{wall_id}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"      Diagnostic PNG saved to {output_file}")
