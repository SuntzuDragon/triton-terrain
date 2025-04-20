import numpy as np


def to_obj(array: np.ndarray):
    obj_verts = ''
    obj_faces = ''

    height, width = array.shape

    for y in range(height - 1):
        for x in range(width - 1):
            obj_verts += f'v {x} {y} {array[y, x]}\n'

            v00 = (y * width + x) + 1
            v01 = v00 + width
            v10 = v00 + 1
            v11 = v00 + width + 1

            obj_faces += f'f {v00} {v01} {v10}\n'
            obj_faces += f'f {v01} {v11} {v10}\n'
        obj_verts += f'v {x+1} {y} {array[y, x+1]}\n'

    for x in range(width):
        obj_verts += f'v {x} {height-1} {array[height-1, x]}\n'

    return obj_verts + '\n' + obj_faces
