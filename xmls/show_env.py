from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np

env_file = 'block_world.xml'
model = load_model_from_path(env_file)
sim = MjSim(model, nsubsteps=1)
sim.model.vis.map.znear = 0.02
sim.model.vis.map.zfar = 50.0
viewer = MjViewer(sim)

# sim.model.body_pos[1] = np.array([0.6, 0.6, 4])
print(sim.data.get_joint_qpos('cube_0'))
print(sim.model.body_pos[1])
print(sim.model.geom_pos[0])
sim.reset()
while True:
    # print(sim.model.body_pos[1], sim.data.body_xpos[1])
    sim.step()
    viewer.render()