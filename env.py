from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import numpy as np
import json
import os
import time


class BlockWordEnv:
    def __init__(self, env_file, random_color=None, debug=False):
        self.env_file = env_file
        self.random_color = random_color
        self.debug = debug
        self.model = load_model_from_path(env_file)
        self.sim = MjSim(self.model, nsubsteps=1)
        self.sim.model.vis.map.znear = 0.02
        self.sim.model.vis.map.zfar = 50.0
        self.cube_size = self.sim.model.geom_size[self.model._geom_name2id['cube_0']]
        self.cuboid_size = self.sim.model.geom_size[self.model._geom_name2id['cuboid_0']]
        self.cube_num = len([i for i in self.sim.model.geom_names if 'cube_' in i])
        self.cuboid_num = len([i for i in self.sim.model.geom_names if 'cuboid_' in i])
        print('total cube num:', self.cube_num)
        print('total cuboid num:', self.cuboid_num)
        self.max_num_per_type = max(self.cube_num, self.cuboid_num)
        self.center_bounds = [-0.42, 0.42]#[0, self.cuboid_size[0] * self.max_num_per_type]
        self.pos_candidates = np.arange(self.center_bounds[0],
                                        self.center_bounds[1] + self.cube_size[0],
                                        self.cube_size[0])
        self.modder = TextureModder(self.sim)
        self.cur_id = {'cube': 0,
                       'cuboid': 0}
        self.viewer = None
        if random_color:
            self.reset_viewer()

    def reset(self):
        self.sim.reset()
        self.sim.forward()
        self.cur_id = {'cube': 0,
                       'cuboid': 0}
        if self.random_color:
            self.modder.whiten_materials()
            for name in self.sim.model.geom_names:
                self.modder.rand_all(name)
        if self.viewer is not None:
            self.reset_viewer()

    def render(self):
        if self.viewer is None:
            self.reset_viewer()
        self.viewer.render()

    def reset_viewer(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.lookat[:3] = np.array([0, 0, 0.1])
        self.viewer.cam.distance = 2
        self.viewer.cam.elevation = -20

    def move_block(self, target_pos, bk_type='cube'):
        # center bounds: [0, 0.1 * 30]
        assert bk_type == 'cube' or bk_type == 'cuboid'
        prev_pose = self.sim.data.get_joint_qpos('{0:s}_{1:d}'.format(bk_type, self.cur_id[bk_type]))
        # if self.debug:
        #     print('{0:s}_{1:d} pose before moving:'.format(bk_type, self.cur_id[bk_type]), prev_pose)
        post_pose = prev_pose.copy()
        post_pose[:3] = target_pos
        self.sim.data.set_joint_qpos('{0:s}_{1:d}'.format(bk_type, self.cur_id[bk_type]), post_pose)
        if self.debug:
            print('{0:s}_{1:d} pose after moving:'.format(bk_type, self.cur_id[bk_type]), post_pose)
        self.cur_id[bk_type] += 1


    def randomize_color(self):
        if self.random_color is None:
            return
        assert self.random_color >= 0 and self.random_color <= 1
        ran = np.random.random(1)[0]
        if ran < self.random_color:
            for name in self.sim.model.geom_names:
                self.modder.rand_all(name)

    def gen_ran_bk_configs(self, render=False):
        # prob = np.exp(-0.1 * np.arange(30))
        cuboid_num = np.random.choice(30, 1)[0]
        cube_num = np.random.choice(30, 1)[0]
        total_num = cuboid_num + cube_num
        blocks = [0] * cube_num + [1] * cuboid_num
        permuted_blocks = np.random.permutation(blocks)
        cur_x = self.center_bounds[0]
        layer_num = 1
        for i in range(total_num):
            bk = permuted_blocks[i]
            bk_type = 'cube' if bk == 0 else 'cuboid'
            bk_size = self.cube_size if bk == 0 else self.cuboid_size
            z = (2 * layer_num - 1) * self.cube_size[2]
            y = 0
            if self.center_bounds[1] - cur_x < bk_size[0]:
                layer_num += 1
                cur_x = self.center_bounds[0]
                continue
            else:
                bk_lower_limit = cur_x + bk_size[0]
                pos_candidates = self.pos_candidates[self.pos_candidates >= bk_lower_limit]
                x = np.random.choice(pos_candidates, 1)[0]
                cur_x = x + bk_size[0]
                target_pos = np.array([x, y, z])
                self.move_block(target_pos, bk_type=bk_type)
        self.sim.forward()
        if render:
            self.render()



def main():
    env_file = './xmls/block_world.xml'
    BKWorld = BlockWordEnv(env_file=env_file, debug=True, random_color=True)
    BKWorld.reset()
    BKWorld.render()
    BKWorld.gen_ran_bk_configs(render=True)
    while True:
        # BKWorld.reset()
        # BKWorld.gen_ran_bk_configs(render=True)
        BKWorld.render()

if __name__ == '__main__':
    main()







