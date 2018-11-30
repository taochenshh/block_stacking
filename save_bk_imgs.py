from env.block_world import BlockWordEnv
import cv2
import argparse
import json
import numpy as np
import os

def main():
    np.set_printoptions(precision=4, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_file", help="environment file",
                        type=str, default="./xmls/block_world.xml")
    parser.add_argument("--save_dir", help="dir to save images",
                        type=str, default="../data")
    parser.add_argument("--max_cfgs", help="maximum number of configurations to generate",
                        type=int, default="100000")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    img_folders = [d for d in os.listdir(args.save_dir) if os.path.isdir(os.path.join(args.save_dir, d))]
    start_idx = len(img_folders)
    folder_idx = start_idx
    for idx in range(start_idx, args.max_cfgs):
        BKWorld = BlockWordEnv(env_file=args.env_file, debug=False, random_color=True, random_num=5)
        BKWorld.reset()
        imgs, stable = BKWorld.simulate_one_epoch()
        for j, img in enumerate(imgs):
            save_folder = os.path.join(args.save_dir, '{0:08d}'.format(folder_idx))
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(os.path.join(save_folder, 'img.png'), img)
            # cv2.imwrite(os.path.join(save_folder, 'img_flip.png'), img[:, ::-1])
            with open(os.path.join(save_folder, 'label.json'), 'w') as f:
                json.dump(bool(stable), f, indent=2)
            folder_idx += 1
        del BKWorld


if __name__ == '__main__':
    main()