from mujoco_py import load_model_from_path, MjSim, MjViewer

env_file = 'block_world.xml'
model = load_model_from_path(env_file)
sim = MjSim(model, nsubsteps=1)
viewer = MjViewer(sim)
sim.reset()
sim.step()
while True:
    viewer.render()