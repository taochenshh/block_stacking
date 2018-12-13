'''
Author: Tao Chen (CMU RI)
Date: 11/25/2018
'''
import argparse

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

from bk_dataset import BKDataset


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='logging training status every n iterations')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='save model every n epochs')
    parser.add_argument('--save_dir', type=int, default=50,
                        help='dir to save model and log')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    data_transforms = {
        'full': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dataset = BKDataset(mode='full', transform=data_transforms['full'])

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             drop_last=False)

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    resnet_no_fc = nn.Sequential(*list(model_ft.children())[:-1])
    for param in resnet_no_fc:
        param.requires_grad = False

    model = resnet_no_fc.to(device)
    model.eval()

    image_features = []
    image_labels = []
    for i_batch, (rgbs, labels) in enumerate(tqdm(dataloader)):
        batch_rgbs = rgbs.float().to(device)
        batch_features = torch.squeeze(model(batch_rgbs))
        image_features.append(batch_features.detach().cpu().numpy())
        image_labels.append(labels.detach().cpu().numpy())
    image_features = np.concatenate(image_features, axis=0)
    image_labels = np.concatenate(image_labels, axis=0)
    np.savez('features.npz', features=image_features, labels=image_labels)


if __name__ == '__main__':
    main()
