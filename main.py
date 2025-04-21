import time
import torch
import fire
from perlin_gpu import compute_noise_grid_parallel
from perlin_serial import compute_noise_grid_serial
from obj import save_to_obj
from perlin_gpu import compute_noise_grid_parallel
from perlin_serial import compute_noise_grid_serial
from png import save_to_png
from objrender import render


class Terrain(object):
    def timing(self, iterations=5):
        triton_times = []
        serial_times = []
        grid_sizes = [128, 256, 512, 1024, 2048]
        scale = 85.0
        print("Timing comparison for multiple grid sizes:")
        print("{:<8} {:<20} {:<20}".format(
            "Size", "Triton (GPU)", "Serial (CPU)"))

        for size in grid_sizes:
            for _ in range(iterations):
                # Triton (GPU) version.
                start = time.time()
                _ = compute_noise_grid_parallel(
                    size, size, scale)
                torch.cuda.synchronize()  # ensure GPU work completes
                triton_times.append(time.time() - start)

                # Serial (CPU) version.
                start = time.time()
                _ = compute_noise_grid_serial(
                    size, size, scale)
                serial_times.append(time.time() - start)

            avg_triton = sum(triton_times) / iterations
            avg_serial = sum(serial_times) / iterations
            print("{:<8} {:<20.4f} {:<20.4f}".format(
                size, avg_triton, avg_serial))

    def obj(self, filename: str = 'terrain.obj', size=1024, scale=85.0):
        noise_grid = compute_noise_grid_parallel(size, size, scale)
        obj = save_to_obj(noise_grid)
        if not filename.endswith('.obj'):
            filename = filename + '.obj'
        with open(filename, 'w') as f:
            f.write(obj)
        render(objfilename=filename)

    def png(self, size=1024, scale=85.0, filename='terrain.png'):
        noise_grid = compute_noise_grid_parallel(size, size, scale)
        save_to_png(noise_grid, filename)

    def view(self, filename='terrain.png'):
        render(objfilename=filename)


def main():
    fire.Fire(Terrain)


if __name__ == '__main__':
    main()
