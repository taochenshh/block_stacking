import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


class StabilityChecker:
    def __init__(self, model_dir):
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")
        self.model_dir = model_dir
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        self.model = model_ft.to(self.device)
        print('loading checkpoint...')
        self.load_model(model_dir, self.model)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def eval(self, image):
        self.model.eval()
        image = self.transform(image)
        image = image.to(self.device).float()
        image = image.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(image)
            pred = logits.max(1, keepdim=True)[1]
        return pred.item()

    def load_model(self, save_dir, model):
        ckpt_file = os.path.join(save_dir, 'model_best.pth')
        ckpt = torch.load(ckpt_file, map_location=self.device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v
                           for k, v in ckpt['state_dict'].items()
                           if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save_dir', type=str, default='./data',
                        help='dir to save model and log')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


if __name__ == '__main__':
    main()
