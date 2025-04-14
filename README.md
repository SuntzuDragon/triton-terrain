# Perlin Terrain Generator with Triton

This project uses Triton to compute a height map with a single-layer Perlin noise kernel running on the GPU.

## File Structure

- **main.py:** Main driver script that computes the noise grid.
- **perlin_gpu.py:** Contains the Triton kernel and a helper function to generate the noise grid.
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
