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


# --------------
# Timing Tests
# --------------


def test_grid_size(grid_size, noise_scale, iterations=5):
    triton_times = []
    serial_times = []

    for _ in range(iterations):
        # Triton (GPU) version.
        start = time.time()
        noise_grid = compute(grid_size, grid_size, noise_scale)
        torch.cuda.synchronize()  # ensure GPU work completes
        triton_times.append(time.time() - start)

        # Serial (CPU) version.
        start = time.time()
        _ = compute_noise_grid_serial(grid_size, grid_size, noise_scale)
        serial_times.append(time.time() - start)

    save_to_png(noise_grid, f"triton_{grid_size}")

    avg_triton = sum(triton_times) / iterations
    avg_serial = sum(serial_times) / iterations
    return avg_triton, avg_serial


def save_to_png(noise_grid, name):
    plt.figure(figsize=(6, 6))
    plt.imshow(noise_grid, cmap='viridis', interpolation='lanczos')
    plt.colorbar(label='Noise Value')
    plt.title("Perlin Noise Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(f'{name}.png')
    # print(f"Plot saved as {name}.png")


def main():
    # List of grid sizes to test.
    # grid_sizes = [128, 256, 512, 1024, 2048]
    grid_sizes = [128, 256, 512, 1024]  # demo
    noise_scale = 20.0
    iterations = 5

    print("Timing comparison for multiple grid sizes:")
    print("{:<8} {:<20} {:<20}".format("Size", "Triton (GPU)", "Serial (CPU)"))
    for size in grid_sizes:
        avg_triton, avg_serial = test_grid_size(size, noise_scale, iterations)
        print("{:<8} {:<20.4f} {:<20.4f}".format(size, avg_triton, avg_serial))


if __name__ == '__main__':
    main()
