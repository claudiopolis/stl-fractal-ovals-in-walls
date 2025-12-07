"""
Command-line interface for the wallholer tool.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np

from wallholer.config import Config, load_config, interactive_config
from wallholer.stl_loader import load_stl
from wallholer.wall_detector import detect_walls, Wall
from wallholer.visualizer import visualize_walls
from wallholer.hole_generator import generate_hole_pattern
from wallholer.mesh_modifier import apply_holes_to_walls


def apply_inter_wall_clearance(walls: List[Wall], config: Config) -> List[Wall]:
    """
    Shrink each wall's bounds to maintain min_wall_width clearance from adjacent walls.

    This ensures that ovals placed on one wall won't intrude into adjacent walls.
    The existing boundary checking logic in hole_generator.py will automatically
    enforce the clipped bounds.

    Args:
        walls: List of detected walls
        config: Configuration containing min_wall_width

    Returns:
        List of walls with adjusted bounds
    """
    min_clearance = config.min_wall_width

    def get_primary_axis(normal: np.ndarray) -> int:
        """Get the axis (0=X, 1=Y, 2=Z) that the normal is primarily aligned with."""
        abs_normal = np.abs(normal)
        return int(np.argmax(abs_normal))

    for i, wall in enumerate(walls):
        original_bounds = wall.bounds.copy()

        for j, other_wall in enumerate(walls):
            if i == j:
                continue

            # Determine which axis each wall's normal aligns with
            wall_axis = get_primary_axis(wall.normal)
            other_axis = get_primary_axis(other_wall.normal)

            # Skip if walls are parallel (same primary axis)
            # Parallel walls are separated along their normal and don't interfere
            if wall_axis == other_axis:
                continue

            # Get the two perpendicular axes for overlap checking
            perp_axes = [0, 1, 2]
            perp_axes.remove(wall_axis)

            # Check if walls overlap in the perpendicular dimensions
            overlaps_perp = True
            for axis in perp_axes:
                if (wall.bounds[1][axis] < other_wall.bounds[0][axis] or
                    wall.bounds[0][axis] > other_wall.bounds[1][axis]):
                    overlaps_perp = False
                    break

            if not overlaps_perp:
                continue  # Walls don't overlap in perpendicular axes, not adjacent

            # Walls overlap in perpendicular dimensions and are perpendicular
            # Check proximity and apply clearance along BOTH axes:
            # 1. wall_axis (wall's normal direction)
            # 2. other_axis (other wall's normal direction)
            # This is needed because perpendicular walls can be adjacent along either axis
            # Note: other_wall.bounds already includes its thickness, so we only add min_clearance
            clearance_needed = min_clearance

            # Check proximity along wall's normal axis (wall_axis)
            # Check if other_wall is on the "low side" of wall (before wall's minimum)
            # other_wall.bounds[1] is other_wall's max, wall.bounds[0] is wall's min
            if other_wall.bounds[1][wall_axis] <= wall.bounds[0][wall_axis]:
                # Other wall is at/below wall's minimum face
                gap = wall.bounds[0][wall_axis] - other_wall.bounds[1][wall_axis]
                if gap < clearance_needed:
                    # Gap is insufficient, shrink wall's minimum bound
                    new_min = other_wall.bounds[1][wall_axis] + clearance_needed
                    wall.bounds[0][wall_axis] = new_min

            # Check if other_wall is on the "high side" of wall (after wall's maximum)
            # other_wall.bounds[0] is other_wall's min, wall.bounds[1] is wall's max
            if other_wall.bounds[0][wall_axis] >= wall.bounds[1][wall_axis]:
                # Other wall is at/above wall's maximum face
                gap = other_wall.bounds[0][wall_axis] - wall.bounds[1][wall_axis]
                if gap < clearance_needed:
                    # Gap is insufficient, shrink wall's maximum bound
                    new_max = other_wall.bounds[0][wall_axis] - clearance_needed
                    wall.bounds[1][wall_axis] = new_max

            # Also check proximity along other wall's normal axis (other_axis)
            # This catches cases where walls are adjacent along the other direction
            # Example: vertical wall (X-normal) meeting horizontal bottom wall (Z-normal)
            # are adjacent along Z, not X

            # Check if wall's minimum overlaps with or is close to other_wall's range
            # This handles cases where walls are flush or overlapping
            if (wall.bounds[0][other_axis] >= other_wall.bounds[0][other_axis] and
                wall.bounds[0][other_axis] <= other_wall.bounds[1][other_axis] + clearance_needed):
                # Wall's minimum is within or near other_wall's range
                # Shrink wall's minimum to clear other_wall
                new_min = other_wall.bounds[1][other_axis] + clearance_needed
                wall.bounds[0][other_axis] = new_min

            # Check if wall's maximum overlaps with or is close to other_wall's range
            if (wall.bounds[1][other_axis] >= other_wall.bounds[0][other_axis] - clearance_needed and
                wall.bounds[1][other_axis] <= other_wall.bounds[1][other_axis]):
                # Wall's maximum is within or near other_wall's range
                # Shrink wall's maximum to clear other_wall
                new_max = other_wall.bounds[0][other_axis] - clearance_needed
                wall.bounds[1][other_axis] = new_max

        # Validate bounds are still valid (min < max on all axes)
        if np.any(wall.bounds[0] >= wall.bounds[1]):
            print(f"Warning: Wall {wall.id} bounds collapsed after inter-wall clearance adjustment")
            print(f"         Restoring original bounds. This wall may be too constrained for holes.")
            wall.bounds = original_bounds

    return walls


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Add oval hole patterns to thin walls in 3D models (STL format)"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input STL file path"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="YAML configuration file (if not provided, will use interactive mode)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output STL file path (default: input_modified.stl)"
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_modified.stl"

    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = load_config(args.config)
    else:
        print("No configuration file provided. Starting interactive mode...")
        config = interactive_config()

    # Load STL file
    print(f"\nLoading STL file: {input_path}")
    mesh = load_stl(str(input_path))

    # Detect walls
    print("\nDetecting thin flat walls...")
    walls = detect_walls(mesh, config)
    print(f"Found {len(walls)} potential walls")

    if len(walls) == 0:
        print("No walls detected. Exiting.")
        sys.exit(0)

    # Visualize walls if in interactive mode
    if not args.config:
        print("\nGenerating visualization...")
        visualize_walls(mesh, walls, config)
        print("Visualization saved to 'wall_visualization.png'")

        # Get user selection of walls to modify
        wall_ids = input("\nEnter wall IDs to modify (comma-separated, e.g., 1,3,5): ")
        selected_walls = [int(w.strip()) for w in wall_ids.split(",")]

        bottom_wall = int(input("Enter the wall ID that represents the bottom (build plate): "))
    else:
        # Use configuration
        selected_walls = config.selected_walls
        bottom_wall = config.bottom_wall

        # Still generate visualization for reference
        print("\nGenerating visualization for reference...")
        visualize_walls(mesh, walls, config)
        print("Visualization saved to 'wall_visualization.png'")

    # Apply inter-wall clearance to prevent ovals from intruding into adjacent walls
    print("\nApplying inter-wall clearance constraints...")
    walls = apply_inter_wall_clearance(walls, config)

    # Generate hole patterns
    print(f"\nGenerating oval hole patterns for {len(selected_walls)} wall(s)...")
    hole_patterns = {}
    for wall_id in selected_walls:
        if wall_id < len(walls):
            patterns = generate_hole_pattern(walls[wall_id], config, bottom_wall, mesh)
            hole_patterns[wall_id] = patterns
            print(f"  Wall {wall_id}: {len(patterns)} ovals generated")

    # Apply holes to mesh
    print("\nApplying holes to mesh...")
    modified_mesh = apply_holes_to_walls(mesh, walls, hole_patterns, config)

    # Save output
    print(f"\nSaving modified STL to: {output_path}")
    modified_mesh.export(str(output_path))

    print("\nDone!")


if __name__ == "__main__":
    main()
