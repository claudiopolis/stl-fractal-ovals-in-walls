"""
STL file loading functionality.
"""

import trimesh
import numpy as np


def load_stl(file_path: str) -> trimesh.Trimesh:
    """
    Load an STL file and return a trimesh object.

    Args:
        file_path: Path to the STL file

    Returns:
        A trimesh.Trimesh object representing the 3D model
    """
    mesh = trimesh.load(file_path, force='mesh')

    # Ensure the mesh is valid
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load {file_path} as a valid mesh")

    # Fix normals if needed (requires networkx)
    try:
        mesh.fix_normals()
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Warning: Could not fix normals (missing dependency: {e}). Continuing anyway...")

    # Make sure mesh is watertight if possible
    if not mesh.is_watertight:
        print(f"Warning: Mesh is not watertight. Attempting to repair...")

        # First, detect open boundaries and cap them manually
        print(f"  Detecting open boundaries...")
        try:
            # Get all edges and find which appear only once (boundary edges)
            from collections import defaultdict
            edge_count = defaultdict(int)

            for face in mesh.faces:
                # Each face has 3 edges
                for i in range(3):
                    v1, v2 = face[i], face[(i+1) % 3]
                    # Use sorted tuple as key for undirected edge
                    edge = tuple(sorted([v1, v2]))
                    edge_count[edge] += 1

            # Boundary edges appear exactly once
            boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

            # Debug: show edge count distribution
            edge_counts = {}
            for count in edge_count.values():
                edge_counts[count] = edge_counts.get(count, 0) + 1
            print(f"  Edge count distribution: {edge_counts}")
            print(f"  Found {len(boundary_edges)} boundary edges")

            if boundary_edges:
                print(f"  Adding cap to close the mesh...")

                # Get all boundary vertices
                boundary_verts = set()
                for edge in boundary_edges:
                    boundary_verts.update(edge)

                # Find the boundary loop (assumes single continuous loop)
                boundary_coords = mesh.vertices[list(boundary_verts)]

                # Simple fan triangulation from centroid
                centroid = boundary_coords.mean(axis=0)
                centroid_idx = len(mesh.vertices)

                # Add centroid vertex
                mesh.vertices = np.vstack([mesh.vertices, [centroid]])

                # Create faces connecting boundary edges to centroid
                new_faces = []
                for edge in boundary_edges:
                    v1, v2 = edge
                    new_faces.append([v1, v2, centroid_idx])

                if new_faces:
                    mesh.faces = np.vstack([mesh.faces, new_faces])

                    # Clean up
                    mesh.merge_vertices()
                    mesh.remove_degenerate_faces()

                    print(f"  Added {len(new_faces)} triangles to close boundary")

        except Exception as e:
            print(f"  Error closing boundaries: {e}")

        # Check for non-manifold edges
        if not mesh.is_watertight and edge_counts.get(1, 0) == 0:
            # No boundary edges but still not watertight = non-manifold mesh
            print(f"  ")
            print(f"  ERROR: Your STL file has non-manifold geometry!")
            print(f"  ")
            print(f"  This typically happens with hollow models where inner and outer")
            print(f"  surfaces share edges. Boolean operations require manifold meshes.")
            print(f"  ")
            print(f"  Solutions:")
            print(f"  1. Repair your STL in CAD software (e.g., Blender, Mes")
            print(f"     Mesh > Cleanup > Make Manifold)")
            print(f"  2. Use an online STL repair service (e.g., 3D Maker Noob)")
            print(f"  3. Simplify your model to have solid walls instead of hollow")
            print(f"  ")
            print(f"  The program will attempt to proceed, but boolean operations")
            print(f"  will likely fail.")
            print(f"  ")

        if mesh.is_watertight:
            print(f"  ✓ Mesh repaired and is now watertight")
        else:
            print(f"  Warning: Mesh still not watertight after repair.")
            print(f"  For boolean operations to work, creating a simplified manifold version...")
            try:
                # Last resort: voxelize and remesh
                pitch = (mesh.bounds[1] - mesh.bounds[0]).min() / 50  # 50 voxels on smallest dimension
                voxelized = mesh.voxelized(pitch=pitch)
                mesh = voxelized.marching_cubes
                print(f"  ✓ Created voxelized manifold mesh (simplified geometry)")
            except Exception as e:
                print(f"  Error: Could not create manifold mesh: {e}")

    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Bounding box: {mesh.bounds}")
    print(f"Mesh properties: watertight={mesh.is_watertight}, volume={mesh.is_volume}, winding_consistent={mesh.is_winding_consistent}")

    return mesh
