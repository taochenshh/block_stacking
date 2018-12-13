from mujoco_py import load_model_from_path, MjSim, MjViewer

env_file = 'block_world.xml'
model = load_model_from_path(env_file)
sim = MjSim(model, nsubsteps=1)
sim.model.vis.map.znear = 0.02
sim.model.vis.map.zfar = 50.0
viewer = MjViewer(sim)

# sim.model.body_pos[1] = np.array([0.6, 0.6, 4])
sim.reset()
cube_pose = sim.data.get_joint_qpos('cube_0')
print(cube_pose)
# cube_pose[:3] += 0.1
sim.data.set_joint_qpos('cube_0', cube_pose)
sim.forward()
cube_pose = sim.data.get_joint_qpos('cube_0')
print(cube_pose)
print(sim.model.body_pos[1])
print(sim.model.geom_pos[0])

while True:
    # print(sim.model.body_pos[1], sim.data.body_xpos[1])
    sim.step()
    viewer.render()
