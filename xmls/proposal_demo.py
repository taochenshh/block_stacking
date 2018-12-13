import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder

env_file = 'block_world.xml'
model = load_model_from_path(env_file)
sim = MjSim(model, nsubsteps=1)
sim.model.vis.map.znear = 0.02
sim.model.vis.map.zfar = 50.0
modder = TextureModder(sim)
viewer = MjViewer(sim)
viewer.cam.lookat[:3] = np.array([0, 0, 0.1])
viewer.cam.distance = 2
viewer.cam.elevation = -20
# sim.model.body_pos[1] = np.array([0.6, 0.6, 4])
sim.reset()
cube_pose = sim.data.get_joint_qpos('cube_0')
cube_pose[:3] = np.array([0, 0, 0.02])
sim.data.set_joint_qpos('cube_0', cube_pose)

cube_pose = sim.data.get_joint_qpos('cuboid_0')
cube_pose[:3] = np.array([0., 0, 0.06])
sim.data.set_joint_qpos('cuboid_0', cube_pose)

cube_pose = sim.data.get_joint_qpos('cube_1')
cube_pose[:3] = np.array([0.12, 0, 0.02])
sim.data.set_joint_qpos('cube_1', cube_pose)

cube_pose = sim.data.get_joint_qpos('cube_2')
cube_pose[:3] = np.array([0.08, 0, 0.10])
sim.data.set_joint_qpos('cube_2', cube_pose)

cube_pose = sim.data.get_joint_qpos('cube_3')
cube_pose[:3] = np.array([-0.08, 0, 0.10])
sim.data.set_joint_qpos('cube_3', cube_pose)
sim.forward()
modder.whiten_materials()  # ensures materials won't impact colors
while True:
    sim.step()
    viewer.render()

    for name in sim.model.geom_names:
        print(name)
        modder.rand_all(name)
