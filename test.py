import time
import torch
from perlin_gpu import compute_noise_grid_parallel
from perlin_serial import compute_noise_grid_serial
from png import save_to_png

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
