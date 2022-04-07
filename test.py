import os
import time

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from process_data import autoaugment
from model_generator import *

torch.set_printoptions(profile="full")


if __name__ == "__main__":
    # 加载pt模型
    model = mobile_former_151(100, pre_train=True, state_dir="./acc/mobile_former_151.pth")
    model.cuda()
    model.eval()

    num_correct = 0
    num_samples = 0
    total_correct = 0
    total_samples = 0

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_aug = transforms.Compose([
        autoaugment.CIFAR10Policy(),
        transforms.Resize(224),
        transforms.ToTensor(),
        # 接收tensor
        transforms.RandomErasing(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    cifar_val = torchvision.datasets.CIFAR100('./dataset/', train=False, download=True, transform=transform)
    # cifar_val_aug = torchvision.datasets.CIFAR100('./dataset/', train=False, download=True, transform=transform_aug)
    # cifar_val += cifar_val_aug

    loader_val = DataLoader(cifar_val, batch_size=64, shuffle=True, pin_memory=True)
    print(len(cifar_val))

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    with torch.no_grad():
        for x, y in loader_val:
            x = x.to(torch.device('cuda'), dtype=torch.float32)
            y = y.to(torch.device('cuda'), dtype=torch.long)
            scores = model(x)
            # _,是batch_size*概率，preds是batch_size*最大概率的列号
            _, preds = scores.max(1)
            num_correct = (preds == y).sum()
            num_samples = preds.size(0)
            total_correct += num_correct
            total_samples += num_samples
            print(float(total_correct) / total_samples)

        acc = float(total_correct) / total_samples
    print(acc)

    image_PIL = Image.open('./test_img/orange.jpg')
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image_tensor = transform(image_PIL)
    # 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
    image_tensor.unsqueeze_(0)
    print(image_tensor.shape)
    image_tensor = image_tensor.cuda()

    starttime = time.time()
    out = model(image_tensor)
    endtime = time.time()
    print(int(round((endtime - starttime) * 1000)))
    print(out.shape)
    # 得到预测结果，并且从大到小排序
    _, preds = out.max(1)
    print(preds)
