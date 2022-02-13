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
from model import MobileFormer
from utils.config import config
from utils.utils import cutmix, cutmix_criterion, RandomErasing


def check_accuracy(loader, model, device=None, dtype=None):
    num_correct = 0
    num_samples = 0
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
        mixup=0,
        criterion=nn.CrossEntropyLoss(),
        model=None, loader_train=None,
        loader_val=None, scheduler=None, optimizer=None, wd=None,
        epochs=1, device=None, dtype=None, check_point_dir=None, save_epochs=None, mode=None
):
    model = model.to(device)
    acc = 0
    accs = [0]
    losses = []

    record_dir_acc = check_point_dir + 'record_val_acc.npy'
    record_dir_loss = check_point_dir + 'record_loss.npy'
    model_save_dir = check_point_dir + 'mobile_former_151_100.pth'
    model.load_state_dict(torch.load(model_save_dir))

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
            # 300个iteration就计算一下测试集准确率
            if t % 200 == 0:
                print("Iteration:" + str(t) + ', average Loss = ' + str(loss_value))
        # 每个epoch记录一次测试集准确率和所有batch的平均训练损失
        model.eval()
        total_loss /= t
        losses.append(total_loss)

        acc = check_accuracy(loader_val, model, device=device)
        accs.append(np.array(acc))
        print("Epoch:" + str(e) + ', Val acc = ' + str(acc) + ', average Loss = ' + str(total_loss))
        # 如果到了保存的epoch或者是训练完成的最后一个epoch
        if (mode == 'run' and e % save_epochs == 0 and e != 0) or (mode == 'run' and e == epochs - 1):
            np.save(record_dir_acc, np.array(accs))
            np.save(record_dir_loss, np.array(losses))
            torch.save(model.state_dict(), model_save_dir)
    return acc


def run(
        mixup=0,
        criterion=nn.CrossEntropyLoss(),
        mode='run', model=None,
        search_epoch=None, lr_range=None, wd_range=[-4, -2],
        search_result_save_dir=None,
        run_epoch=30, lr=0.0008, wd=0.10,
        check_point_dir=None, save_epochs=None,
        T_mult=None, loader_train=None, loader_val=None, device=None, dtype=None
):
    if mode == 'search':
        num_iter = 10000000
        epochs = search_epoch
    else:
        num_iter = 1
        epochs = run_epoch
    if mode == 'search':
        print('Searching under lr: 10 ** (', lr_range[0], ',', lr_range[1], ') , wd: 10 ** (', wd_range[0], ',',
              wd_range[1], '), every ', epochs, ' epoch')
    else:
        print(mode + 'ing under lr: ' + str(lr) + ' , wd: ' + str(wd) + ' for ', str(epochs), ' epochs.')
    for i in range(num_iter):
        model_ = model
        if mode == 'search':
            # low: 采样下界，float类型，默认值为0
            # high: 采样上界，float类型，默认值为1
            # size: 输出样本数目，为int或元组(tuple)类型，例如，size=(m,n,k), 则输出m*n*k个样本，缺省时输出1个值
            # 返回值：ndarray类型，其形状和参数size中描述一致
            learning_rate = 10 ** np.random.uniform(lr_range[0], lr_range[1])
            weight_decay = 10 ** np.random.uniform(wd_range[0], wd_range[1])
        else:
            learning_rate = lr
            weight_decay = wd

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                      betas=(0.9, 0.999), weight_decay=wd)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=T_mult)
        args = {
            'mixup': mixup,
            'criterion': criterion,
            'model': model_, 'loader_train': loader_train, 'loader_val': loader_val,
            'scheduler': lr_scheduler, 'optimizer': optimizer, 'wd': wd,
            'epochs': epochs, 'device': device, 'dtype': dtype,
            'check_point_dir': check_point_dir, 'save_epochs': save_epochs, 'mode': mode
        }
        print('#############################     Training...     #############################')
        val_acc = train(**args)
        # 最后一个epoch的最后一次测试集准确率
        print('Training for ' + str(epochs) + ' epochs, learning rate: ', learning_rate, ', weight decay: ',
              weight_decay, ', Val acc: ', val_acc)

        if mode == 'search':
            with open(search_result_save_dir + 'search_result.txt', "a") as f:
                f.write(str(epochs) + ' epochs, learning rate:' + str(learning_rate) + ', weight decay: ' + str(
                    weight_decay) + ', Val acc: ' + str(val_acc) + '\n')
        if mode == 'run':
            print('Done, check_point saved in ', check_point_dir)


if __name__ == '__main__':
    print('###############################  Training test  ###############################')
    dtype = torch.float32
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Device: ', device)
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

    cfg = config['mf151']
    model = MobileFormer(cfg)

    print('###############################  Dataset loaded  ##############################')
    args = {
        'loader_train': loader_train, 'loader_val': loader_val,
        'device': device, 'dtype': dtype,
        # Basic setting, mode: 'run' or 'search'
        'mode': 'run',
        # 'model': mobile_former_151(10, pre_train=True, state_dir='./check_point/mobile_former_151_100.pth'),
        # 'model': model,
        'model': model,
        'criterion': nn.CrossEntropyLoss(),
        'mixup': 0,
        # 余弦退火
        'T_mult': 2,
        # If search: (Masked if run)
        'search_epoch': 2, 'lr_range': [-4, -2.5], 'wd_range': [-3, -1],
        'search_result_save_dir': './search_result/',
        # If run: (Masked if search)
        'run_epoch': 450, 'lr': 0.0009, 'wd': 0.10,
        'check_point_dir': './check_point/',
        'save_epochs': 3,
    }
    run(**args)
