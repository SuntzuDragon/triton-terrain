import numpy as np
from matplotlib import pyplot as plt


def save_to_png(noise_grid: np.ndarray, filename: str):
    plt.figure(figsize=(6, 6))
    plt.imshow(noise_grid, cmap='viridis', interpolation='lanczos')
    plt.colorbar(label='Noise Value')
    plt.title("Perlin Noise Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    if not filename.endswith('.png'):
        filename = filename + '.png'
    plt.savefig(filename)
