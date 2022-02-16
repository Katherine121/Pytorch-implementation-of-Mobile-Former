import os

import torch
import numpy as np
from torch.optim import lr_scheduler, optimizer
from torch.serialization import save
from torchvision.transforms.transforms import Scale

import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

from torchvision.transforms import autoaugment
# from torchvision.transforms import RandomErasing
from model_generator import *
from utils.utils import cutmix, cutmix_criterion, RandomErasing


def check_accuracy(loader, model, device=None, dtype=None):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        t = 0
        for x, y in loader:
            x = x.to(device, dtype=dtype)
            y = y.to(device, dtype=torch.long)
            scores = model(x)
            # _,是batch_size*概率，preds是batch_size*最大概率的列号
            _, preds = scores.max(1)
            num_correct = (preds == y).sum()
            num_samples = preds.size(0)
            total_correct += num_correct
            total_samples += num_samples

            if t % 100 == 0:
                print('预测正确的图片数目' + str(num_correct))
                print('总共的图片数目' + str(num_samples))
            t += 1
        acc = float(total_correct) / total_samples
    return acc


def train(
        loader_train=None, loader_val=None,
        device=None, dtype=None,
        model=None,
        criterion=nn.CrossEntropyLoss(),
        scheduler=None, optimizer=None,
        epochs=450, check_point_dir=None, save_epochs=None
):
    acc = 0
    accs = [0]
    losses = []

    record_dir_acc = check_point_dir + 'record_val_acc.npy'
    record_dir_loss = check_point_dir + 'record_loss.npy'
    model_save_dir = check_point_dir + 'mobile_former_151_100.pth'

    model = model.to(device)

    for e in range(epochs):
        model.train()
        total_loss = 0
        for t, (x, y) in enumerate(loader_train):
            x = x.to(device=device, dtype=dtype, non_blocking=True)
            y = y.to(device=device, dtype=torch.long, non_blocking=True)
            # 原x+混x，原t，混t，原混比
            inputs, targets_a, targets_b, lam = cutmix(x, y, 1)
            # inputs, targets_a, targets_b = map(Variable, (inputs,
            #                                               targets_a, targets_b))
            # 原x+混x->原y+混y
            outputs = model(inputs)

            # 原y+混y和原t，混t求损失
            # 2
            # loss = Mixup.mixup_criterion(criterion, scores, targets_a, targets_b, lam)
            loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss_value = np.array(loss.item())
            total_loss += loss_value

            # 1
            optimizer.zero_grad()
            # 3
            loss.backward()
            # optimizer.param_groups： 是长度为2的list，其中的元素是2个字典
            # optimizer.param_groups[0]： 长度为6的字典，包括[‘amsgrad', ‘params', ‘lr', ‘betas', ‘weight_decay', ‘eps']这6个参数；
            # optimizer.param_groups[1]： 表示优化器的状态的一个字典
            # for group in optimizer.param_groups:  # Adam-W
            #     for param in group['params']:
            #         # -weight decay*learning rate
            #         param.data = param.data.add(param.data, alpha=-wd * group['lr'])
            # 4
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            # 200个iteration就计算一下测试集准确率
            if t % 200 == 0:
                print("Iteration:" + str(t) + ', average Loss = ' + str(loss_value))

        total_loss /= t
        losses.append(total_loss)

        acc = check_accuracy(loader_val, model, device=device)
        accs.append(np.array(acc))

        # 每个epoch记录一次测试集准确率和所有batch的平均训练损失
        print("Epoch:" + str(e) + ', Val acc = ' + str(acc) + ', average Loss = ' + str(total_loss))
        # 如果到了保存的epoch或者是训练完成的最后一个epoch
        if (e % save_epochs == 0 and e != 0) or e == epochs - 1:
            np.save(record_dir_acc, np.array(accs))
            np.save(record_dir_loss, np.array(losses))
            torch.save(model.state_dict(), model_save_dir)
    return acc


def run(
        loader_train=None, loader_val=None,
        device=None, dtype=None,
        model=None,
        criterion=nn.CrossEntropyLoss(),
        T_mult=2,
        epoch=450, lr=0.0009, wd=0.10,
        check_point_dir=None, save_epochs=3,

):
    epochs = epoch
    model_ = model
    learning_rate = lr
    weight_decay = wd
    print('Training under lr: ' + str(lr) + ' , wd: ' + str(wd) + ' for ', str(epochs), ' epochs.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  betas=(0.9, 0.999), weight_decay=wd)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=T_mult)
    args = {
        'loader_train': loader_train, 'loader_val': loader_val,
        'device': device, 'dtype': dtype,
        'model': model_,
        'criterion': criterion,
        'scheduler': lr_scheduler, 'optimizer': optimizer,
        'epochs': epochs,
        'check_point_dir': check_point_dir, 'save_epochs': save_epochs,
    }
    print('#############################     Training...     #############################')
    val_acc = train(**args)
    # 最后一个epoch的最后一次测试集准确率
    print('Training for ' + str(epochs) + ' epochs, learning rate: ', learning_rate, ', weight decay: ',
          weight_decay, ', Val acc: ', val_acc)
    print('Done, model saved in ', check_point_dir)


if __name__ == '__main__':
    print('############################### Dataset loading ###############################')

    transform = transforms.Compose([
        transforms.Lambda(autoaugment.RandAugment(num_ops=2, magnitude=10)),
        transforms.Resize(224),
        # RandomErasing(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # 取出50000作为训练集
    cifar_train = dset.CIFAR100('./dataset/', train=True, download=True, transform=transform)
    loader_train = DataLoader(cifar_train, batch_size=128, shuffle=True, pin_memory=True)

    # 10000作为测试集
    cifar_val = dset.CIFAR100('./dataset/', train=False, download=True, transform=transform)
    loader_val = DataLoader(cifar_val, batch_size=64, shuffle=True, pin_memory=True)

    # imagenet_train = dset.ImageFolder(root='/datasets/ILSVRC2012/train/', transform=transform)
    # loader_train = DataLoader(imagenet_train, batch_size=256, shuffle=True, num_workers=4)
    # print(len(imagenet_train))
    # imagenet_val = dset.ImageFolder(root='./val/', transform=transform)
    # loader_val = DataLoader(imagenet_val, batch_size=128, shuffle=True, num_workers=4)
    # print(len(imagenet_val))

    print('###############################  Dataset loaded  ##############################')

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device('cuda')
    args = {
        'loader_train': loader_train, 'loader_val': loader_val,
        'device': device, 'dtype': torch.float32,
        # 'model': mobile_former_151(100),
        'model': mobile_former_151(100, pre_train=True, state_dir='./check_point/mobile_former_151_100.pth'),
        # 'model': MobileFormer(cfg),
        'criterion': nn.CrossEntropyLoss(),
        # 余弦退火
        'T_mult': 2,
        'epoch': 450, 'lr': 0.0009, 'wd': 0.10,
        'check_point_dir': './check_point/', 'save_epochs': 3,
    }
    run(**args)
