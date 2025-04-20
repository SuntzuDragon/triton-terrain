import time
from matplotlib import pyplot as plt
import torch
from perlin_gpu import compute_noise_grid_parallel
from perlin_serial import compute_noise_grid_serial
from obj import to_obj


# --------------
# Timing Tests
# --------------


def test_grid_size(grid_size, noise_scale, iterations=5):
    triton_times = []
    serial_times = []

    for _ in range(iterations):
        # Triton (GPU) version.
        start = time.time()
        noise_grid = compute_noise_grid_parallel(
            grid_size, grid_size, noise_scale)
        torch.cuda.synchronize()  # ensure GPU work completes
        triton_times.append(time.time() - start)

        # Serial (CPU) version.
        start = time.time()
        noise_serial = compute_noise_grid_serial(
            grid_size, grid_size, noise_scale)
        serial_times.append(time.time() - start)

    save_to_png(noise_grid, f"triton_{grid_size}")
    # Generate obj here

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
    # grid_sizes = [64, 128, 256, 512]  # demo
    # noise_scale = 20.0
    # iterations = 5

    # print("Timing comparison for multiple grid sizes:")
    # print("{:<8} {:<20} {:<20}".format("Size", "Triton (GPU)", "Serial (CPU)"))
    # for size in grid_sizes:
    #     avg_triton, avg_serial = test_grid_size(size, noise_scale, iterations)
    #     print("{:<8} {:<20.4f} {:<20.4f}".format(size, avg_triton, avg_serial))
    resolution = 1024
    noise_grid = compute_noise_grid_parallel(resolution, resolution, 85.0)
    attempt = 23
    save_to_png(noise_grid, f'octaves-attempt{attempt}')
    obj = to_obj(noise_grid)
    with open(f'bobj-octave{attempt}.obj', 'w') as f:
        f.write(obj)


if __name__ == '__main__':
    main()
