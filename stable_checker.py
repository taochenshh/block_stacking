import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import os
from bk_dataset import BKDataset
import logger
import shutil

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    total_loss /= float(len(train_loader.dataset))
    acc = correct / float(len(train_loader.dataset))
    logger.logkv('epoch', epoch)
    logger.logkv('train/loss', total_loss)
    logger.logkv('train/acc', acc)
    logger.dumpkvs()
    return total_loss, acc

def test(args, model, device, test_loader, epoch=None, val=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).long()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = float(correct) / len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * acc))
    if val:
        logger.logkv('epoch', epoch)
        logger.logkv('val/loss', test_loss)
        logger.logkv('val/acc', acc)
        logger.dumpkvs()
    return test_loss, acc


def save_model(save_dir, model, epoch, is_best):
    data = {'epoch': epoch,
            'state_dict': model.state_dict()}
    ckpt_file = os.path.join(save_dir, 'ckpt_{:08d}.pth'.format(epoch))
    torch.save(data, ckpt_file)
    if is_best:
        shutil.copyfile(ckpt_file, os.path.join(save_dir, 'model_best.pth'))


def load_model(save_dir, model):
    ckpt_file = os.path.join(save_dir, 'model_best.pth')
    ckpt = torch.load(ckpt_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test', action='store_true', help='test the model with test data')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
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
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = BKDataset(mode='train', transform=data_transforms['train'])
    val_dataset = BKDataset(mode='val', transform=data_transforms['val'])
    test_dataset = BKDataset(mode='test', transform=data_transforms['test'])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.test_batch_size,
                                             shuffle=True,
                                             num_workers=4)
    if args.test:
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.test_batch_size,
                                                  shuffle=True,
                                                  num_workers=4)

    log_dir = os.path.join(args.save_dir, 'logs')
    model_dir = os.path.join(args.save_dir, 'model')
    if not args.test:
        logger.configure(dir=log_dir, format_strs=['tensorboard', 'stdout'])

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model = model_ft.to(device)

    if args.test:
        print('loading checkpoint...')
        load_model(model_dir, model)
        test_loss, test_acc = test(args, model, device, test_loader, val=False)
        print('Test accuracy:', test_acc)
        print('Test loss:', test_loss)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_acc = -np.inf
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch)
            test_loss, test_acc = test(args, model, device, val_loader, epoch, val=True)
            if epoch % args.save_interval == 0:
                is_best = False
                if test_acc > best_acc:
                    best_acc = test_acc
                    is_best = True
                save_model(model_dir, model, epoch, is_best)


if __name__ == '__main__':
    main()