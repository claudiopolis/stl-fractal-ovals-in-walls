"""
Configuration management for wallholer.
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class Config:
    """Configuration for wall hole generation."""

    # Wall detection parameters
    wall_thickness_threshold: float = 5.0  # mm
    wall_aspect_ratio: float = 5.0  # minimum ratio of long/short dimensions

    # Oval hole parameters
    max_oval_length_z: float = 10.0  # mm, maximum length along z-axis
    oval_aspect_ratio: int = 3  # ratio of length to width (whole number)
    min_wall_width: float = 1.0  # mm, minimum material width between holes and edges
    num_oval_sizes: int = 3  # number of different oval sizes

    # User selections (for config file mode)
    selected_walls: List[int] = field(default_factory=list)
    bottom_wall: int = 0


def load_config(config_path: str) -> Config:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    return Config(
        wall_thickness_threshold=data.get('wall_thickness_threshold', 5.0),
        wall_aspect_ratio=data.get('wall_aspect_ratio', 5.0),
        max_oval_length_z=data.get('max_oval_length_z', 10.0),
        oval_aspect_ratio=data.get('oval_aspect_ratio', 3),
        min_wall_width=data.get('min_wall_width', 1.0),
        num_oval_sizes=data.get('num_oval_sizes', 3),
        selected_walls=data.get('selected_walls', []),
        bottom_wall=data.get('bottom_wall', 0)
    )


def interactive_config() -> Config:
    """Create configuration through interactive prompts."""
    config = Config()

    print("\n=== Wall Detection Parameters ===")
    print("(Walls must be THINNER than threshold and FLATTER than aspect ratio)")
    response = input(f"Maximum wall thickness in mm (walls must be thinner than this) (default: {config.wall_thickness_threshold}): ")
    if response.strip():
        config.wall_thickness_threshold = float(response)

    response = input(f"Minimum wall aspect ratio (long/short dimension ratio, e.g. 5 = 5:1) (default: {config.wall_aspect_ratio}): ")
    if response.strip():
        config.wall_aspect_ratio = float(response)

    print("\n=== Oval Hole Parameters ===")
    response = input(f"Maximum oval length along z-axis in mm (default: {config.max_oval_length_z}): ")
    if response.strip():
        config.max_oval_length_z = float(response)

    response = input(f"Oval aspect ratio (length/width, whole number) (default: {config.oval_aspect_ratio}): ")
    if response.strip():
        config.oval_aspect_ratio = int(response)

    response = input(f"Minimum wall width between holes and edges in mm (default: {config.min_wall_width}): ")
    if response.strip():
        config.min_wall_width = float(response)

    response = input(f"Number of different oval sizes (default: {config.num_oval_sizes}): ")
    if response.strip():
        config.num_oval_sizes = int(response)

    return config


def save_example_config(output_path: str = "example_config.yaml"):
    """Save an example configuration file."""
    example = {
        'wall_thickness_threshold': 5.0,
        'wall_aspect_ratio': 5.0,
        'max_oval_length_z': 10.0,
        'oval_aspect_ratio': 3,
        'min_wall_width': 1.0,
        'num_oval_sizes': 3,
        'selected_walls': [0, 1, 2],
        'bottom_wall': 0
    }

    with open(output_path, 'w') as f:
        yaml.dump(example, f, default_flow_style=False, sort_keys=False)
