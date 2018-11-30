from stability_checker import StabilityChecker
from env.block_world import BlockWordEnv
import argparse
import numpy as np
import cv2


def main():
    np.set_printoptions(precision=4, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_file", help="environment file",
                        type=str, default="./xmls/block_world.xml")
    parser.add_argument("--model_dir", help="neural network model",
                        type=str, default="./data/model")
    args = parser.parse_args()
    test_configs = [{'cube_0': [0, 0, 0.02],
                     'cuboid_0': [0., 0, 0.06],
                     'cube_1': [0.08, 0, 0.14],
                     'cube_2': [0.08, 0, 0.10],
                     'cube_3': [-0.08, 0, 0.10]}]
    test_id = 0
    SC = StabilityChecker(model_dir=args.model_dir)
    BKWorld = BlockWordEnv(env_file=args.env_file,
                           debug=False,
                           random_color=False,
                           random_num=5)
    pred_right = []
    test_num = 100
    for i in range(test_num):
        BKWorld.reset()
        BKWorld.move_given_blocks(test_configs[test_id])
        BKWorld.step()
        img = BKWorld.get_img()
        label = BKWorld.check_stability(render=False)
        print('[Ground Truth] stable ?:', label)
        cv2.imwrite('test.png', img)
        pred = SC.eval(img)
        print('[Prediction] stable ?:', pred)
        pred_right.append(pred == int(label))
    print('Accuracy in %d tests: %f' % (test_num, np.mean(pred_right)))


if __name__ == '__main__':
    main()