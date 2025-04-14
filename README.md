# Perlin Terrain Generator with Triton

This project uses Triton to compute a height map with a single-layer Perlin noise kernel running on the GPU. The generated noise grid is then used to build a 3D mesh. Each vertex is colored based on height—low vertices appear green and high vertices appear red—and the mesh is exported as a Wavefront OBJ file.

## File Structure

- **main.py:** Main driver script that computes the noise grid, generates the mesh, and exports the OBJ file.
- **perlin_triton.py:** Contains the Triton kernel and a helper function to generate the noise grid.
- **requirements.txt:** Required Python packages.
- **README.md:** This file.

## Setup and Usage

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the main script:
    ```bash
    python main.py
    ```

After running, an OBJ file named terrain.obj will be created in the project directory.