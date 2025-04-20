from perlin_gpu import compute_noise_grid_parallel
from perlin_serial import compute_noise_grid_serial
from obj import save_to_obj
from png import save_to_png
from test import test_grid_size


def main():
    # List of grid sizes to test.
    grid_sizes = [128, 256, 512, 1024, 2048]
    grid_sizes = [64, 128, 256, 512]  # demo
    noise_scale = 20.0
    iterations = 5

    print("Timing comparison for multiple grid sizes:")
    print("{:<8} {:<20} {:<20}".format("Size", "Triton (GPU)", "Serial (CPU)"))
    for size in grid_sizes:
        avg_triton, avg_serial = test_grid_size(size, noise_scale, iterations)
        print("{:<8} {:<20.4f} {:<20.4f}".format(size, avg_triton, avg_serial))
    resolution = 256
    noise_grid = compute_noise_grid_parallel(resolution, resolution, 85.0)
    attempt = 24

    save_to_png(noise_grid, f'octaves-attempt{attempt}')
    obj = save_to_obj(noise_grid)
    with open(f'bobj-octave{attempt}.obj', 'w') as f:
        f.write(obj)

    resolution = 256
    noise_grid = compute_noise_grid_serial(resolution, resolution, 85.0)
    save_to_png(noise_grid, f'octaves-attempt-serial{attempt}')
    obj = save_to_obj(noise_grid)
    with open(f'bobj-octave-serial{attempt}.obj', 'w') as f:
        f.write(obj)


if __name__ == '__main__':
    main()
