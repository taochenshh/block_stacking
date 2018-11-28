import numpy as np

data = np.load('features.npz')
features = data['features']
labels = data['labels']
print('num positive:', np.sum(labels > 0))
print('num negative:', np.sum(labels == 0))
a = 1