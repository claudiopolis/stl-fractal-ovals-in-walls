"""
Visualization of detected walls using PyVista.
"""

import numpy as np
import pyvista as pv
import trimesh
from typing import List
import matplotlib.pyplot as plt

from wallholer.wall_detector import Wall
from wallholer.config import Config


def visualize_walls(mesh: trimesh.Trimesh, walls: List[Wall], config: Config, output_file: str = "wall_visualization.png"):
    """
    Create a visualization of the mesh with numbered walls.

    Args:
        mesh: The original mesh
        walls: List of detected walls
        config: Configuration
        output_file: Path to save the PNG visualization
    """
    # Create a PyVista plotter (off-screen for PNG generation)
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])

    # Convert trimesh to pyvista
    vertices = mesh.vertices
    faces = np.hstack([[3] + list(face) for face in mesh.faces])
    pv_mesh = pv.PolyData(vertices, faces)

    # Add the main mesh in light gray
    plotter.add_mesh(pv_mesh, color='lightgray', opacity=0.3, show_edges=True)

    # Color palette for walls
    colors = [
        'red', 'blue', 'green', 'yellow', 'cyan', 'magenta',
        'orange', 'purple', 'pink', 'lime', 'brown', 'navy'
    ]

    # Highlight each wall with a different color and add label
    for i, wall in enumerate(walls):
        color = colors[i % len(colors)]

        # Create a mesh for just this wall
        wall_faces = mesh.faces[wall.faces]
        wall_face_array = np.hstack([[3] + list(face) for face in wall_faces])
        wall_mesh = pv.PolyData(vertices, wall_face_array)

        # Add wall mesh with color
        plotter.add_mesh(wall_mesh, color=color, opacity=0.7, show_edges=False)

        # Add label at wall center
        label_text = f"{wall.id}"
        plotter.add_point_labels(
            [wall.center],
            [label_text],
            font_size=36,
            point_size=20,
            text_color='white',
            font_family='arial',
            shape_color=color,
            shape_opacity=1.0,
            render_points_as_spheres=True,
            always_visible=True,
            bold=True
        )

    # Set camera position for good view
    plotter.camera_position = 'iso'
    plotter.show_axes()

    # Save screenshot
    plotter.screenshot(output_file)
    plotter.close()

    # Also create 2D projections for each wall
    create_wall_projections(mesh, walls, "wall_projections.png")


def create_wall_projections(mesh: trimesh.Trimesh, walls: List[Wall], output_file: str = "wall_projections.png"):
    """
    Create 2D projection images for each wall.

    Args:
        mesh: The original mesh
        walls: List of detected walls
        output_file: Path to save the PNG with all projections
    """
    if len(walls) == 0:
        return

    # Calculate grid size for subplots
    n_walls = len(walls)
    cols = min(3, n_walls)
    rows = (n_walls + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_walls == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, wall in enumerate(walls):
        ax = axes[i]

        # Get wall vertices and edges
        vertices_2d, edges_2d = project_wall_to_2d_with_edges(mesh, wall)

        # Plot the wall outline
        if len(vertices_2d) > 0:
            # Draw the edges (outline)
            for edge in edges_2d:
                ax.plot(edge[:, 0], edge[:, 1], 'b-', linewidth=2)

            # Also fill the convex hull to show the wall area
            from scipy.spatial import ConvexHull
            if len(vertices_2d) >= 3:
                try:
                    hull = ConvexHull(vertices_2d)
                    for simplex in hull.simplices:
                        ax.plot(vertices_2d[simplex, 0], vertices_2d[simplex, 1], 'b-', linewidth=1, alpha=0.5)
                    ax.fill(vertices_2d[hull.vertices, 0], vertices_2d[hull.vertices, 1], 'lightblue', alpha=0.3)
                except:
                    # Fallback: just scatter the vertices
                    ax.scatter(vertices_2d[:, 0], vertices_2d[:, 1], c='blue', s=10)

            # Set equal aspect ratio
            ax.set_aspect('equal')
            ax.set_title(f'Wall {wall.id}\n'
                        f'Dims: {wall.dimensions[0]:.1f} x {wall.dimensions[1]:.1f} x {wall.dimensions[2]:.1f} mm\n'
                        f'Thickness: {wall.thickness:.1f} mm')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('mm')
            ax.set_ylabel('mm')

    # Hide unused subplots
    for i in range(n_walls, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def project_wall_to_2d(wall: Wall) -> np.ndarray:
    """
    Project a wall onto its best-fit 2D plane.

    Args:
        wall: The wall to project

    Returns:
        2D coordinates of the projected vertices
    """
    vertices = wall.vertices
    normal = wall.normal

    # Find the two axes perpendicular to the normal
    # Choose axes that give the largest projection
    abs_normal = np.abs(normal)
    min_component = np.argmin(abs_normal)

    # Create two perpendicular vectors in the plane
    if min_component == 0:
        u = np.array([0, -normal[2], normal[1]])
    elif min_component == 1:
        u = np.array([-normal[2], 0, normal[0]])
    else:
        u = np.array([-normal[1], normal[0], 0])

    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # Project vertices onto the 2D plane
    vertices_2d = np.column_stack([
        np.dot(vertices, u),
        np.dot(vertices, v)
    ])

    return vertices_2d


def project_wall_to_2d_with_edges(mesh: trimesh.Trimesh, wall: Wall):
    """
    Project a wall onto its best-fit 2D plane and extract edges.

    Args:
        mesh: The original mesh
        wall: The wall to project

    Returns:
        Tuple of (vertices_2d, edges_2d) where edges_2d is a list of line segments
    """
    normal = wall.normal

    # Find the two axes perpendicular to the normal
    abs_normal = np.abs(normal)
    min_component = np.argmin(abs_normal)

    # Create two perpendicular vectors in the plane
    if min_component == 0:
        u = np.array([0, -normal[2], normal[1]])
    elif min_component == 1:
        u = np.array([-normal[2], 0, normal[0]])
    else:
        u = np.array([-normal[1], normal[0], 0])

    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # Get all vertices from this wall's faces
    wall_faces = mesh.faces[wall.faces]

    # Project all mesh vertices that belong to this wall
    all_vertices = mesh.vertices
    vertices_2d_all = np.column_stack([
        np.dot(all_vertices, u),
        np.dot(all_vertices, v)
    ])

    # Get unique vertices from wall faces
    unique_vertex_indices = np.unique(wall_faces.flatten())
    vertices_2d = vertices_2d_all[unique_vertex_indices]

    # Extract edges from the faces
    edges_2d = []
    for face in wall_faces:
        # Each face has 3 edges
        for i in range(3):
            v1_idx = face[i]
            v2_idx = face[(i + 1) % 3]
            edge = np.array([vertices_2d_all[v1_idx], vertices_2d_all[v2_idx]])
            edges_2d.append(edge)

    return vertices_2d, edges_2d
