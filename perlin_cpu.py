# Serial implementation referenced from https://adrianb.io/2014/08/09/perlinnoise.html

import math
import random

# Create a permutation list with 256 unique values and duplicate it
p = list(range(256))
random.shuffle(p)
p += p  # Duplicate so we can index with wrap-around


def fade(t):
    """
    Fade function as defined by Ken Perlin.
    This eases coordinate values so that they will "smoothly" interpolate.
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a, b, t):
    """
    Linearly interpolate between a and b with weight t.
    """
    return a + t * (b - a)


def grad(hash, x, y):
    """
    Given a hash value, returns a gradient vector and calculates
    the dot product with (x, y). For simplicity, only 4 vectors are used:
    (1,1), (-1,1), (1,-1), (-1,-1).
    """
    h = hash & 3  # Use the lower 2 bits to pick one of the 4 directions.
    if h == 0:
        return x + y
    elif h == 1:
        return -x + y
    elif h == 2:
        return x - y
    else:  # h == 3
        return -x - y


def perlin(x, y):
    """
    Generate 2D Perlin noise for coordinates (x, y).
    """
    # Determine grid cell coordinates
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255

    # Compute relative x, y (the fractional part)
    xf = x - math.floor(x)
    yf = y - math.floor(y)

    # Compute the fade curves for x and y
    u = fade(xf)
    v = fade(yf)

    # Hash coordinates of the 4 square corners
    aa = p[p[xi] + yi]
    ab = p[p[xi] + yi + 1]
    ba = p[p[xi + 1] + yi]
    bb = p[p[xi + 1] + yi + 1]

    # Calculate the dot product between the gradient and the distance vector for each corner
    x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1, yf), u)
    x2 = lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u)

    # Interpolate the two results along y
    return lerp(x1, x2, v)


# Example usage: Print Perlin noise values for a 2D grid
if __name__ == "__main__":
    for i in range(10):
        for j in range(10):
            x = i / 5.0
            y = j / 5.0
            print(f"perlin({x:.2f}, {y:.2f}) = {perlin(x, y):.4f}")
        print()
