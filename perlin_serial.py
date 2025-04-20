import random
import numpy as np
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


def compute_noise_grid_serial(width, height, scale=85.0):
    random.seed(42)
    perm = list(range(256))  # [0, 1, .. 255]
    random.shuffle(perm)
    # duplicate the permutation table (first half == second half, doubled in length)
    perm = perm + perm
    OCTAVES = 4

    noise_grid = np.empty((height, width), dtype=np.float32)
    for j in range(height):
        for i in range(width):
            octave_result = 0.0
            amplitude = 1.0
            frequency = 1.0
            norm = 0.0
            for o in range(OCTAVES):
                x = i / scale
                y = j / scale
                result = perlin_serial(x*frequency, y*frequency, perm)

                octave_result += result * amplitude
                norm += amplitude
                amplitude *= 0.4  # persistance
                frequency *= 2  # lacunarity

            octave_result = octave_result / norm

            octave_result = (octave_result + 1) / 2
            octave_result *= octave_result * octave_result
            octave_result *= 100
            noise_grid[j, i] = octave_result

    return noise_grid
