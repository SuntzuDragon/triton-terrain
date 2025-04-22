import numpy as np


def save_to_obj(array: np.ndarray) -> str:
    height, width = array.shape

    # Generate all vertex lines
    verts = [f'v {x} {y} {array[y, x]}'
             for y in range(height)
             for x in range(width)]

    faces = []
    # Calculate width for indexing
    for y in range(height - 1):
        row_offset = y * width
        for x in range(width - 1):
            v00 = row_offset + x + 1
            v01 = v00 + width
            v10 = v00 + 1
            v11 = v01 + 1
            # faces.append(f'f {v00} {v01} {v10}')
            # faces.append(f'f {v01} {v11} {v10}')
            faces.append(f'f {v00} {v10} {v01}')
            faces.append(f'f {v01} {v10} {v11}')

    # Join into single string
    return "\n".join(verts + faces) + "\n"
