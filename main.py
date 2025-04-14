import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from perlin_gpu import compute
import random

# --------------------------
# Serial CPU Implementation
# --------------------------


def fade_serial(t):
    # 6t^5 - 15^4 + 10^3
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp_serial(a, b, t):
    return a + t * (b - a)


def grad_serial(hash_val, x_val, y_val):
    h = hash_val & 3
    if h == 0:
        return x_val + y_val
    elif h == 1:
        return -x_val + y_val
    elif h == 2:
        return x_val - y_val
    else:
        return -x_val - y_val


def perlin_serial(x, y, perm):
    xi = int(x) % 256
    yi = int(y) % 256
    xf = x - int(x)  # 0.4857...
    yf = y - int(y)
    u = fade_serial(xf)  # float portion of x & y smoothed
    v = fade_serial(yf)

    aa = perm[perm[xi] + yi]
    ab = perm[perm[xi] + yi + 1]
    ba = perm[perm[xi + 1] + yi]
    bb = perm[perm[xi + 1] + yi + 1]

    x1 = lerp_serial(grad_serial(aa, xf, yf), grad_serial(ba, xf - 1, yf), u)
    x2 = lerp_serial(grad_serial(ab, xf, yf - 1),
                     grad_serial(bb, xf - 1, yf - 1), u)
    return lerp_serial(x1, x2, v)


def compute_noise_grid_serial(width, height, scale):
    random.seed(42)
    perm = list(range(256))  # [0, 1, .. 255]
    random.shuffle(perm)
    # duplicate the permutation table (first half == second half, doubled in length)
    perm = perm + perm

    noise_grid = np.empty((height, width), dtype=np.float32)
    for j in range(height):
        for i in range(width):
            x = i / scale
            y = j / scale
            noise_grid[j, i] = perlin_serial(x, y, perm)
    return noise_grid

# ------------------------
# Terrain Mesh Generation
# ------------------------


def generate_terrain(noise_grid):
    height, width = noise_grid.shape
    vertices = []
    for j in range(height):
        for i in range(width):
            # x: i, y: noise height, z: j.
            h = noise_grid[j, i]
            vertices.append((i, h, j))

    h_vals = [v[1] for v in vertices]
    min_h = min(h_vals)
    max_h = max(h_vals)

    colored_vertices = []
    for v in vertices:
        x, h, z = v
        t = (h - min_h) / (max_h - min_h) if max_h != min_h else 0.0
        # Interpolate from green (low) to red (high).
        r = t
        g = 1.0 - t
        b = 0.0
        colored_vertices.append((x, h, z, r, g, b))

    # Generate faces (two triangles per cell).
    faces = []
    for j in range(height - 1):
        for i in range(width - 1):
            idx = j * width + i + 1  # OBJ is 1-indexed
            v1 = idx
            v2 = idx + width
            v3 = idx + width + 1
            v4 = idx + 1
            faces.append((v1, v2, v3))
            faces.append((v1, v3, v4))
    return colored_vertices, faces


def write_obj(filename, vertices, faces):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write("v {} {} {} {} {} {}\n".format(
                v[0], v[1], v[2], v[3], v[4], v[5]))
        for face in faces:
            f.write("f {} {} {}\n".format(face[0], face[1], face[2]))

# --------------
# Timing Tests
# --------------


def test_grid_size(grid_size, noise_scale, iterations=5):
    triton_times = []
    serial_times = []

    for _ in range(iterations):
        # Triton (GPU) version.
        start = time.time()
        _ = compute(grid_size, grid_size, noise_scale)
        torch.cuda.synchronize()  # ensure GPU work completes
        triton_times.append(time.time() - start)

        # Serial (CPU) version.
        start = time.time()
        _ = compute_noise_grid_serial(grid_size, grid_size, noise_scale)
        serial_times.append(time.time() - start)

    avg_triton = sum(triton_times) / iterations
    avg_serial = sum(serial_times) / iterations
    return avg_triton, avg_serial


def main():
    # List of grid sizes to test.
    grid_sizes = [512, 1024, 2048, 4096]
    noise_scale = 20.0
    iterations = 5

    print("Timing comparison for multiple grid sizes:")
    print("{:<8} {:<20} {:<20}".format("Size", "Triton (GPU)", "Serial (CPU)"))
    for size in grid_sizes:
        avg_triton, avg_serial = test_grid_size(size, noise_scale, iterations)
        print("{:<8} {:<20.4f} {:<20.4f}".format(size, avg_triton, avg_serial))

    # Generate final terrain mesh using the Triton version from the largest grid.
    final_size = grid_sizes[-1]
    print("\nGenerating terrain mesh for grid size {}x{}...".format(
        final_size, final_size))
    noise_grid = compute(final_size, final_size, noise_scale)

    plt.figure(figsize=(6, 6))
    plt.imshow(noise_grid, cmap='viridis', interpolation='lanczos')
    plt.colorbar(label='Noise Value')
    plt.title("Perlin Noise Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig('perlin_noise.png')
    print("Plot saved as perlin_noise.png")

    vertices, faces = generate_terrain(noise_grid)

    output_obj = "terrain.obj"
    write_obj(output_obj, vertices, faces)
    print("OBJ file exported to:", output_obj)


if __name__ == '__main__':
    main()
