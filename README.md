# 3D Wall-Holer

A Python tool for automatically adding decorative oval hole patterns to thin walls in 3D models (STL format). Designed for FDM 3D printing applications where you want to reduce material usage and print time while maintaining structural integrity.

## Features

- **Automatic Wall Detection**: Intelligently identifies thin flat walls in your 3D model
- **Interactive Visualization**: Generates PNG renders showing detected walls with ID numbers
- **Flexible Interface**: Supports both interactive mode and configuration file mode
- **Fractal Hole Patterns**: Creates aesthetically pleasing patterns with multiple oval sizes
- **Z-Axis Alignment**: Orients oval holes perpendicular to the build plate for optimal FDM printing
- **Diagonal Edge Handling**: Correctly handles walls with diagonal cuts/edges
- **Wavelength Optimization**: Level 0 ovals optimized for maximum size with exact edge clearances
- **Diagnostic Output**: Generates JSON and PNG files showing oval placement with unique IDs
- **Configurable Parameters**: Full control over wall detection, hole size, spacing, and pattern complexity

## Installation

1. Clone or download this repository
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Additional Requirements

For boolean operations, you'll need either:
- **Blender** (recommended): Install from https://www.blender.org/
- **OpenSCAD**: Install from https://openscad.org/

Make sure the chosen tool is available in your system PATH.

## Usage

### Interactive Mode

Run without a configuration file to use interactive prompts:

```bash
python -m wallholer input.stl
```

The program will:
1. Ask for wall detection and hole parameters
2. Detect walls and generate visualization images
3. Prompt you to select which walls to modify
4. Generate holes and save the modified STL

### Configuration File Mode

Run with a YAML configuration file for automated processing:

```bash
python -m wallholer input.stl --config my_config.yaml
```

You can also specify a custom output path:

```bash
python -m wallholer input.stl --config my_config.yaml --output modified.stl
```

### Example STL files

Visit this thing at [Thingiverse](https://www.thingiverse.com/thing:7226225) to see a sample STL file before and after.

### Example Workflow

1. **First run (interactive)** to explore your model:
   ```bash
   python -m wallholer my_model.stl
   ```

2. **Review visualizations**:
   - `wall_visualization.png`: 3D render with numbered walls
   - `wall_projections.png`: 2D projections of each detected wall

3. **Create configuration** based on wall IDs you want to modify:
   ```bash
   cp example_config.yaml my_config.yaml
   # Edit my_config.yaml to set selected_walls and other parameters
   ```

4. **Run with configuration** for final output:
   ```bash
   python -m wallholer my_model.stl --config my_config.yaml --output my_model_holed.stl
   ```

## Configuration Parameters

### Wall Detection

- `wall_thickness_threshold` (default: 6.0): Maximum thickness in mm for a wall to be considered "thin"
- `min_wall_dimension` (default: 20.0): Minimum width and length for a wall to be detected

### Hole Pattern

- `max_oval_length_z` (default: 70.0): Maximum length of the largest ovals along the z-axis in mm (Level 0 ovals will be optimized to fit wall with exact edge clearances)
- `oval_aspect_ratio` (default: 3): Ratio of oval length to width (e.g., 3 means 3:1 ratio)
- `min_wall_width` (default: 4.0): Minimum wall material width between holes and edges in mm
- `num_oval_sizes` (default: 2): Number of different oval sizes for the fractal pattern (2-4 recommended)

### Wall Selection

- `selected_walls`: List of wall IDs to modify (e.g., [0, 1, 2])
- `bottom_wall`: ID of the wall touching the build plate (determines z-axis orientation)

## Output Files

- `{output}.stl`: The modified 3D model with holes (default: `{input}-holed.stl`)
- `wall_visualization.png`: 3D visualization showing all detected walls with ID numbers
- `wall_projections.png`: 2D projection views of each detected wall with dimensions
- `diagnostic_wall_{id}.png`: Per-wall diagnostic showing oval placement with unique IDs
- `diagnostic_wall_{id}.json`: Per-wall diagnostic data with oval positions and sizes

## Important: STL File Requirements

**Your STL file must be a manifold (watertight) mesh** for boolean operations to work. Non-manifold meshes (common in hollow models where inner and outer surfaces share edges) will cause boolean operations to fail.

### Checking if your STL is manifold:
- The program will automatically detect and warn you if your mesh is non-manifold
- Look for the message: "ERROR: Your STL file has non-manifold geometry!"

### Fixing non-manifold STL files:

**Option 1: Blender**
1. Open your STL in Blender
2. Select the mesh in Edit mode (Tab)
3. Mesh menu > Cleanup > Make Manifold
4. Export as STL

**Option 2: Online STL Repair Services**
- Microsoft 3D Model Repair (free)
- Netfabb Cloud (free tier available)

**Option 3: CAD Software**
- Regenerate your hollow box model ensuring inner and outer surfaces don't share edges
- Use solid modeling instead of surface modeling

## Tips for Best Results

1. **Wall Detection**: If walls aren't detected correctly, adjust `wall_thickness_threshold` (default 6mm)
2. **Hole Spacing**: Ensure `min_wall_width` is appropriate for your printer and material (4mm recommended for FDM)
3. **Pattern Density**: Adjust `num_oval_sizes` for fractal complexity (2-3 works well, higher = more processing time)
4. **Z-Axis Alignment**: Make sure to correctly identify the `bottom_wall` so ovals are oriented vertically
5. **Print Settings**: The modified model may need supports depending on wall orientation
6. **Diagonal Edges**: The tool automatically handles walls with diagonal cuts and maintains proper clearances

## Troubleshooting

### Boolean Operation Fails
- Ensure Blender or OpenSCAD is installed and in your PATH
- Try reducing the number of ovals (increase spacing or reduce sizes)
- Check that your input STL is a valid, watertight mesh

### Walls Not Detected
- Try adjusting `wall_thickness_threshold` (increase if walls are thicker than 6mm)
- Ensure your model actually has thin flat regions that meet minimum size requirements
- Check that walls are larger than `min_wall_dimension` (default 20mm)

### Holes Too Dense or Too Sparse
- Adjust `min_wall_width` to control spacing
- Adjust `max_oval_length_z` to change hole sizes
- Reduce `num_oval_sizes` for simpler patterns

## Requirements

- Python 3.10+
- NumPy
- PyVista (for 3D visualization)
- trimesh (for STL processing)
- PyYAML (for configuration files)
- matplotlib (for 2D projections)
- scipy (for geometric calculations)
- Blender or OpenSCAD (for boolean operations)

## License

This project is provided as-is for educational and personal use.

## Utility Scripts

See [UTILITIES.md](UTILITIES.md) for documentation on additional diagnostic and visualization scripts included in this repository.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.
