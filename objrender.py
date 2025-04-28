import trimesh
import numpy as np
from trimesh.transformations import rotation_matrix
from pyglet import gl


def render(objfilename):
    # load + orient
    scene = trimesh.load(objfilename, force='scene')
    scene.apply_transform(rotation_matrix(-np.pi/2, [1, 0, 0]))
    mesh = next(iter(scene.geometry.values()))
    mesh.apply_translation(-mesh.centroid)

    # height -> colour
    zs = mesh.vertices[:, 2]
    z_norm = (zs - zs.min()) / np.ptp(zs)

    # water, sand, grass, rock, snow
    cols = np.zeros((len(z_norm), 4), dtype=np.uint8)
    cols[z_norm < 0.10] = [70, 130, 180, 255]   # water
    cols[(z_norm >= 0.10) & (z_norm < 0.12)] = [194, 178, 128, 255]   # sand
    cols[(z_norm >= 0.12) & (z_norm < 0.40)] = [34, 139,  34, 255]   # grass
    cols[(z_norm >= 0.30) & (z_norm < 0.70)] = [100, 100, 100, 255]   # rock
    cols[z_norm >= 0.70] = [255, 255, 255, 255]   # snow

    mesh.visual.vertex_colors = cols

    # camera planes
    scene.camera.z_near = 0.01
    scene.camera.z_far = 10_000.0

    # GL setup callback: culling + sky
    def gl_setup(_):
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        # light coming from front‑right‑above
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION,
                     (gl.GLfloat * 4)(1.0, 1.0, 2.0, 0.0))

    center = mesh.centroid
    radius = mesh.extents.max()
    angle = np.radians(-35)

    scene.set_camera(
        # rotate around X by +45
        angles=(np.radians(-25), np.radians(225), 0),
        center=center,          # look at mesh center
        distance=radius*1.5         # back up far enough to see it all
    )

    # show at high FPS
    scene.show(
        callback=gl_setup,
        callback_period=1/60,
        window_kwargs={'vsync': False},
        smooth=False
    )
