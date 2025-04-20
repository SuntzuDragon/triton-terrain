import torch
import triton
import triton.language as tl

# Parallel GPU Implementation


@triton.jit
def fade(t):
    # 6t^5 - 15t^4 + 10t^3 OR t * t * t * (t * (t * 6 - 15) + 10)
    return t * t * t * (t * (t * 6 - 15) + 10)


@triton.jit
def lerp(a, b, t):
    return a + t * (b - a)


@triton.jit
def grad(hash_val, x_val, y_val):
    h = hash_val & 3
    res0 = x_val + y_val
    res1 = -x_val + y_val
    res2 = x_val - y_val
    res3 = -x_val - y_val
    return tl.where(h == 0, res0,
                    tl.where(h == 1, res1,
                             tl.where(h == 2, res2, res3)))


@triton.jit
def perlin(output, perm, scale, width, height, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    i = i[None, :]  # reshaping i to be x dir
    j = j[:, None]  # reshaping j to be y dir

    x_o = tl.cast(i, dtype=tl.float32) / scale
    y_o = tl.cast(j, dtype=tl.float32) / scale

    octave_result = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    amplitude = 1.0
    frequency = 1.0
    norm = 0.0

    OCTAVES: tl.constexpr = 4
    for o in range(OCTAVES):
        x = x_o * frequency
        y = y_o * frequency

        xi = tl.cast(x, dtype=tl.int32) % 256
        yi = tl.cast(y, dtype=tl.int32) % 256

        xf = x - tl.cast(tl.cast(x, dtype=tl.int32), dtype=tl.float32)
        yf = y - tl.cast(tl.cast(y, dtype=tl.int32), dtype=tl.float32)

        u = fade(xf)  # float portion of x & y smoothed
        v = fade(yf)

        xi_0 = tl.load(perm + xi)
        xi_1 = tl.load(perm + xi + 1)
        aa = tl.load(perm + xi_0 + yi)
        ab = tl.load(perm + xi_0 + yi + 1)
        ba = tl.load(perm + xi_1 + yi)
        bb = tl.load(perm + xi_1 + yi + 1)

        x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1, yf), u)
        x2 = lerp(grad(ab, xf, yf - 1),
                  grad(bb, xf - 1, yf - 1), u)
        result = lerp(x1, x2, v)

        octave_result += result * amplitude
        norm += amplitude
        amplitude *= 0.4 # persistance
        frequency *= 2 # lacunarity

    octave_result = octave_result / norm
    
    octave_result = (octave_result + 1) / 2
    octave_result *= octave_result * octave_result
    octave_result *= 100
    mask = (i < width) and (j < height)
    idx = j * width + i
    tl.store(output + idx, octave_result, mask=mask)


def compute_noise_grid_parallel(width, height, scale=85.0):
    torch.manual_seed(42)
    perm = torch.randperm(256, device='cuda')
    perm = torch.cat((perm, perm))

    noise_grid = torch.empty(
        (height*width), dtype=torch.float32, device='cuda')

    BLOCK_SIZE = 16

    def grid(meta): return (triton.cdiv(
        width, meta['BLOCK_SIZE']), triton.cdiv(height, meta['BLOCK_SIZE']))

    perlin[grid](noise_grid, perm, scale, width, height, BLOCK_SIZE=BLOCK_SIZE)

    noise_grid = noise_grid.reshape(height, width).cpu().numpy()

    return noise_grid
