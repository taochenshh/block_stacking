from env.block_world import BlockWordEnv
import argparse
import numpy as np
import cv2


def main():
    np.set_printoptions(precision=4, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_file", help="environment file",
                        type=str, default="./xmls/block_world.xml")
    args = parser.parse_args()
    test_config = {'cube_0': [0, 0, 0.02],
                   'cuboid_0': [0., 0, 0.06],
                   'cube_1': [0.00, 0, 0.14],
                   'cube_2': [0.00, 0, 0.10],
                   'cube_3': [-0.08, 0, 0.10]}

    BKWorld = BlockWordEnv(env_file=args.env_file,
                           debug=True,
                           random_color=False,
                           random_num=5)
    BKWorld.reset()
    BKWorld.move_block_for_demo('cube_0', test_config['cube_0'])
    BKWorld.move_block_for_demo('cuboid_0', test_config['cuboid_0'])
    BKWorld.move_block_for_demo('cube_2', test_config['cube_2'])
    BKWorld.move_block_for_demo('cube_1', test_config['cube_1'])
    while True:
        BKWorld.step()
        BKWorld.render()


if __name__ == '__main__':
    main()