from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import numpy as np
import cv2
import json
import os
import time


class BlockWordEnv:
    def __init__(self, env_file, random_color=None, random_num=5, debug=False):
        self.env_file = env_file
        self.random_color = random_color
        self.random_num = random_num
        self.debug = debug
        self.model = load_model_from_path(env_file)
        self.sim = MjSim(self.model, nsubsteps=1)
        self.sim.model.vis.map.znear = 0.02
        self.sim.model.vis.map.zfar = 50.0
        self.cube_size = self.sim.model.geom_size[self.model._geom_name2id['cube_0']]
        self.cuboid_size = self.sim.model.geom_size[self.model._geom_name2id['cuboid_0']]
        self.cube_num = len([i for i in self.sim.model.geom_names if 'cube_' in i])
        self.cuboid_num = len([i for i in self.sim.model.geom_names if 'cuboid_' in i])
        if self.debug:
            print('total cube num:', self.cube_num)
            print('total cuboid num:', self.cuboid_num)
        self.max_num_per_type = max(self.cube_num, self.cuboid_num)
        self.center_bounds = [-0.25, 0.25]#[0, self.cuboid_size[0] * self.max_num_per_type]
        self.pos_candidates = np.arange(self.center_bounds[0],
                                        self.center_bounds[1] + self.cube_size[0],
                                        self.cube_size[0])
        self.modder = TextureModder(self.sim)
        self.cur_id = {'cube': 0,
                       'cuboid': 0}
        self.viewer = None
        self.active_blocks = []
        if random_color:
            self.reset_viewer()

    def reset(self):
        self.sim.reset()
        self.sim.step()
        self.active_blocks = []
        self.cur_id = {'cube': 0,
                       'cuboid': 0}
        if self.random_color:
            self.randomize_color()
        if self.viewer is not None:
            self.reset_viewer()

    def randomize_color(self):
        self.modder.whiten_materials()
        for name in self.sim.model.geom_names:
            if 'table' in name:
                continue
            self.modder.rand_all(name)

    def simulate_one_epoch(self):
        self.gen_constrained_ran_bk_configs()
        imgs = []
        for i in range(self.random_num):
            img = self.get_img()
            imgs.append(img.copy())
            self.randomize_color()
        stable = self.check_stability(render=False)
        return imgs, stable

    def step(self):
        self.sim.step()

    def forward(self):
        self.sim.forward()

    def get_active_bk_states(self):
        bk_poses = []
        for bk_name in self.active_blocks:
            pose = self.sim.data.get_joint_qpos(bk_name)
            bk_poses.append(pose)
        return np.array(bk_poses)

    def check_stability(self, render=False):
        self.sim.step()
        prev_poses = self.get_active_bk_states()
        for i in range(400):
            self.sim.step()
            if render:
                self.render()
        post_poses = self.get_active_bk_states()
        diff = np.abs(post_poses - prev_poses)
        diff_norm = np.linalg.norm(diff[:, :3], axis=1)
        if self.debug:
            print('prev:')
            print(prev_poses[:, :3])
            print('post')
            print(post_poses[:, :3])
            print('diff norm:', diff_norm)
        stable = np.all(diff_norm < 0.01)
        if self.debug:
            print('Current configuration stable:', stable)
        return stable

    def render(self):
        if self.viewer is None:
            self.reset_viewer()
        self.viewer.render()

    def reset_viewer(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.lookat[:3] = np.array([0, 0, 0.1])
        self.viewer.cam.distance = 2
        self.viewer.cam.elevation = -20

    def move_given_block(self, name, target_pos):
        prev_pose = self.sim.data.get_joint_qpos(name)
        # if self.debug:
        #     print('{0:s}_{1:d} pose before moving:'.format(bk_type, self.cur_id[bk_type]), prev_pose)
        post_pose = prev_pose.copy()
        post_pose[:3] = target_pos
        self.sim.data.set_joint_qpos(name, post_pose)
        self.active_blocks.append(name)
        if self.debug:
            print('{0:s} pose after moving:'.format(name), post_pose)

    def move_given_blocks(self, block_dict):
        for bk_name, pos in block_dict.items():
            self.move_given_block(bk_name, pos)

    def move_block_for_demo(self, name, target_pos):
        prev_pose = self.sim.data.get_joint_qpos(name)
        if self.debug:
            print('{0:s} pose before moving:'.format(name), prev_pose)
        post_pose = prev_pose.copy()
        planned_path = []
        up_steps = 20
        h_steps = 30
        down_steps = 20
        planned_path.append(prev_pose.copy())
        for i in range(up_steps):
            tmp_pose = planned_path[-1].copy()
            tmp_pose[2] += (0.2 - prev_pose[2]) / float(up_steps)
            planned_path.append(tmp_pose.copy())
        for i in range(h_steps + 1):
            tmp_pose = planned_path[-1].copy()
            tmp_pose[0] = (target_pos[0] - prev_pose[0]) * i / h_steps + prev_pose[0]
            tmp_pose[1] = (target_pos[1] - prev_pose[1]) * i / h_steps + prev_pose[1]
            planned_path.append(tmp_pose.copy())
        for i in range(down_steps):
            tmp_pose = planned_path[-1].copy()
            tmp_pose[2] -= (0.2 - target_pos[2]) / float(down_steps)
            planned_path.append(tmp_pose.copy())
        post_pose[:3] = target_pos
        planned_path.append(post_pose.copy())
        for pos in planned_path:
            self.sim.data.set_joint_qpos(name, pos)
            time.sleep(0.02)
            self.sim.forward()
            self.render()
        self.active_blocks.append(name)
        if self.debug:
            print('{0:s} pose after moving:'.format(name), post_pose)

    def move_blocks_for_demo(self, block_dict):
        name = list(block_dict.keys())[0]
        target_pos = block_dict[name]
        initial_poses = {}
        for key in block_dict.keys():
            initial_poses[key] = self.sim.data.get_joint_qpos(key)
        prev_pose = self.sim.data.get_joint_qpos(name)
        if self.debug:
            print('{0:s} pose before moving:'.format(name), prev_pose)
        post_pose = prev_pose.copy()
        planned_delta_path = []
        planned_path = []
        up_steps = 20
        h_steps = 30
        down_steps = 20
        planned_path.append(prev_pose.copy())
        planned_delta_path.append(np.zeros_like(prev_pose))
        for i in range(up_steps):
            tmp_pose = planned_path[-1].copy()
            tmp_pose[2] += (0.2 - prev_pose[2]) / float(up_steps)
            planned_delta_path.append(tmp_pose - planned_path[-1])
            planned_path.append(tmp_pose.copy())
        for i in range(h_steps + 1):
            tmp_pose = planned_path[-1].copy()
            tmp_pose[0] = (target_pos[0] - prev_pose[0]) * i / h_steps + prev_pose[0]
            tmp_pose[1] = (target_pos[1] - prev_pose[1]) * i / h_steps + prev_pose[1]
            planned_delta_path.append(tmp_pose - planned_path[-1])
            planned_path.append(tmp_pose.copy())
        for i in range(down_steps):
            tmp_pose = planned_path[-1].copy()
            tmp_pose[2] -= (0.2 - target_pos[2]) / float(down_steps)
            planned_delta_path.append(tmp_pose - planned_path[-1])
            planned_path.append(tmp_pose.copy())
        post_pose[:3] = target_pos
        planned_delta_path.append(post_pose - planned_path[-1])
        planned_path.append(post_pose.copy())

        for delta_pos in planned_delta_path:
            for bk_name, target_bk_pos in block_dict.items():
                initial_poses[bk_name] = initial_poses[bk_name] + delta_pos
                self.sim.data.set_joint_qpos(bk_name, initial_poses[bk_name])
            time.sleep(0.02)
            self.sim.forward()
            self.render()
        self.active_blocks.append(name)
        if self.debug:
            print('{0:s} pose after moving:'.format(name), self.sim.data.get_joint_qpos(name))


    def move_block(self, target_pos, bk_type='cube'):
        # center bounds: [0, 0.1 * 30]
        assert bk_type == 'cube' or bk_type == 'cuboid'
        bk_name = '{0:s}_{1:d}'.format(bk_type, self.cur_id[bk_type])
        prev_pose = self.sim.data.get_joint_qpos(bk_name)
        # if self.debug:
        #     print('{0:s}_{1:d} pose before moving:'.format(bk_type, self.cur_id[bk_type]), prev_pose)
        post_pose = prev_pose.copy()
        post_pose[:3] = target_pos

        self.sim.data.set_joint_qpos(bk_name, post_pose)
        self.active_blocks.append(bk_name)
        if self.debug:
            print('{0:s}_{1:d} pose after moving:'.format(bk_type, self.cur_id[bk_type]), post_pose)
        self.cur_id[bk_type] += 1

    def get_img(self):
        # return self.get_img_demo()
        img = self.sim.render(camera_name='camera', width=600, height=600, depth=False)
        img = np.flipud(img)
        img = img[:, :, ::-1]
        resized_img = cv2.resize(img[0:500, 50:550], (224, 224), cv2.INTER_AREA)
        # resized_img = cv2.resize(img, (224, 224), cv2.INTER_AREA)
        # cv2.imwrite('test_cut.png', img[0:500, 50:550])
        # cv2.imwrite('test.png', img)
        # cv2.imwrite('test_resize.png', resized_img)
        # cv2.imwrite('test_resize_lr_flip.png', resized_img[:, ::-1, :])
        return resized_img

    def get_img_demo(self):
        img = self.sim.render(camera_name='camera', width=1200, height=700, depth=False)
        img = np.flipud(img)
        img = img[:, :, ::-1]
        return img

    def build_tower(self):
        layer_num = 1
        for i in range(20):
            z = (2 * layer_num - 1) * self.cube_size[2]
            y = 0
            x = 0
            target_pos = np.array([x, y, z])
            self.move_block(target_pos, bk_type='cube')
            layer_num += 1

    def gen_ran_bk_configs(self, render=False):
        # prob = np.exp(-0.1 * np.arange(30))
        while True:
            cuboid_num = np.random.choice(5, 1)[0]
            cube_num = np.random.choice(15, 1)[0]
            if cuboid_num > 0 or cube_num > 0:
                break
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

    def gen_constrained_ran_bk_configs(self, render=False):
        # prob = np.exp(-0.1 * np.arange(30))
        while True:
            cuboid_num = np.random.choice(5, 1)[0]
            cube_num = np.random.choice(15, 1)[0]
            if cuboid_num > 0 or cube_num > 0:
                break
        if self.debug:
            print('Selected cube num:', cube_num)
            print('Selected cuboid num:', cuboid_num)
        total_num = cuboid_num + cube_num
        blocks = [0] * cube_num + [1] * cuboid_num
        permuted_blocks = np.random.permutation(blocks)
        cur_x = self.center_bounds[0]
        layer_num = 1
        layer_pos_candidates = self.pos_candidates.copy()
        filled_segs = []
        for i in range(total_num):
            bk = permuted_blocks[i]
            bk_type = 'cube' if bk == 0 else 'cuboid'
            bk_size = self.cube_size if bk == 0 else self.cuboid_size
            z = (2 * layer_num - 1) * self.cube_size[2]
            y = 0
            bk_lower_limit = cur_x + bk_size[0]
            pos_candidates = layer_pos_candidates[layer_pos_candidates >= bk_lower_limit]
            if pos_candidates.size < 1:
                layer_num += 1
                cur_x = self.center_bounds[0]
                layer_pos_candidates = self.pos_candidates.copy()
                good_ids = np.zeros_like(layer_pos_candidates, dtype=bool)
                for seg in filled_segs:
                    good_ids = np.logical_or(good_ids, np.logical_and(layer_pos_candidates >= seg[0],
                                                                      layer_pos_candidates <= seg[1]))
                layer_pos_candidates = layer_pos_candidates[good_ids]
                if self.debug:
                    print('Layer [{0:d}] pos candidates num: {1:d}'.format(layer_num, layer_pos_candidates.size))
                if layer_pos_candidates.size < 1:
                    break
                filled_segs = []
                continue
            else:
                x = np.random.choice(pos_candidates, 1)[0]
                cur_x = x + bk_size[0]
                target_pos = np.array([x, y, z])
                self.move_block(target_pos, bk_type=bk_type)
                filled_segs.append([x - bk_size[0], cur_x])
        self.sim.forward()
        if render:
            self.render()


def main():
    # np.random.seed(0)
    np.set_printoptions(precision=4, suppress=True)
    env_file = './xmls/block_world.xml'
    BKWorld = BlockWordEnv(env_file=env_file, debug=True, random_color=True)
    BKWorld.reset()
    img, stable = BKWorld.simulate_one_epoch()
    cv2.imwrite('test1.png', img)
    with open('test1.json', 'w') as f:
        json.dump(bool(stable), f, indent=2)

    BKWorld = BlockWordEnv(env_file=env_file, debug=True, random_color=True)
    BKWorld.reset()
    img, stable = BKWorld.simulate_one_epoch()
    cv2.imwrite('test2.png', img)
    with open('test2.json', 'w') as f:
        json.dump(bool(stable), f, indent=2)

    # BKWorld.render()
    # BKWorld.gen_constrained_ran_bk_configs(render=True)
    # BKWorld.check_stability(render=True)
    # while True:
    #     BKWorld.step()
    #     BKWorld.render()

if __name__ == '__main__':
    main()







