import os
from random import shuffle
import json

data_dir = '../data'
img_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
shuffle(img_folders)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 1 - train_ratio - val_ratio
train_num = int(train_ratio * len(img_folders))
val_num = int(val_ratio * len(img_folders))
train_data = img_folders[:train_num]
val_data = img_folders[train_num: train_num + val_num]
test_data = img_folders[train_num + val_num:]
print('train num:', len(train_data))
print('val num:', len(val_data))
print('test num:', len(test_data))
with open('data_split.json', 'w') as f:
    split_idx = {'train': train_data,
                 'val': val_data,
                 'test': test_data,
                 'full': img_folders}
    json.dump(split_idx, f, indent=2)