# Triton Terrain

Triton Terrain is a Python-based project that generates Perlin noise-based terrain using both CPU and GPU implementations. It leverages the Triton library for high-performance GPU computations and provides utilities for rendering and exporting the generated terrain.

## Features

- **Perlin Noise Generation**:
  - Serial (CPU) implementation.
  - Parallel (GPU) implementation using Triton.
- **Export Options**:
  - Save terrain as `.obj` files for 3D rendering.
  - Save terrain as `.png` images for visualization.
- **Rendering**:
  - Render `.obj` files with realistic height-based coloring.
- **Performance Comparison**:
  - Compare execution times between CPU and GPU implementations.

## Requirements

The project requires the following Python libraries:

- `fire==0.7.0`
- `matplotlib==3.10.1`
- `numpy==2.2.5`
- `pyglet==2.1.4`
- `torch==2.6.0`
- `trimesh==4.6.8`
- `triton==3.2.0`

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

The project provides a command-line interface (CLI) powered by `fire`. Below are the available commands:

### 1. Generate and Render Terrain as `.obj`

```bash
python main.py obj --filename terrain.obj --size 1024 --scale 85.0
```

- `filename`: Name of the output `.obj` file (default: `terrain.obj`).
- `size`: Grid size for the terrain (default: `1024`).
- `scale`: Scale factor for Perlin noise (default: `85.0`).

### 2. Save Terrain as `.png`

```bash
python main.py png --filename terrain.png --size 1024 --scale 85.0
```

- `filename`: Name of the output `.png` file (default: `terrain.png`).
- `size`: Grid size for the terrain (default: `1024`).
- `scale`: Scale factor for Perlin noise (default: `85.0`).

### 3. View Rendered Terrain

```bash
python main.py view --filename terrain.obj
```

- `filename`: Name of the `.obj` file to render (default: `terrain.obj`).

### 4. Performance Timing

```bash
python main.py timing --iterations 5
```

- `iterations`: Number of iterations for timing (default: `5`).

## Project Structure

- `main.py`: Entry point for the CLI.
- `perlin_serial.py`: Serial CPU implementation of Perlin noise.
- `perlin_gpu.py`: Parallel GPU implementation of Perlin noise using Triton.
- `obj.py`: Utility to export terrain as `.obj` files.
- `png.py`: Utility to save terrain as `.png` images.
- `objrender.py`: Utility to render `.obj` files with height-based coloring.
- `requirements.txt`: List of required Python libraries.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [Triton](https://github.com/openai/triton) for GPU programming.
- [Trimesh](https://trimsh.org/) for 3D geometry processing.
- [Pyglet](https://pyglet.org/) for rendering support.
