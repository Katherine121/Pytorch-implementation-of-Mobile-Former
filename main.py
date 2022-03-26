import os
import torch.nn as nn
import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from model_generator import *
from process_data import autoaugment
from process_data.utils import cutmix, cutmix_criterion


def prune_loss(model):
    sparsity_loss = 0

    for name, param in model.named_parameters():
        # block.0.former.layers.0.0.fn.to_qkv.weight:torch.Size([192, 192])
        # column
        if 'fn.to_qkv' in name and 'weight' in name:
            qkv = param.data.chunk(3, dim=-1)
            q, k, v = qkv[0], qkv[1], qkv[2]
            # 在列上计算L2正则化
            q_norm = q.norm(2, dim=-1)
            k_norm = k.norm(2, dim=-1)
            v_norm = v.norm(2, dim=-1)
            # 在行上求和
            q1 = q_norm.sum(dim=-1)
            k1 = k_norm.sum(dim=-1)
            v1 = v_norm.sum(dim=-1)
            # 求结构化稀疏度正则项
            q_s = torch.div(q1*q1, (q*q).sum(dim=-1).sum(dim=-1))
            k_s = torch.div(k1*k1, (k*k).sum(dim=-1).sum(dim=-1))
            v_s = torch.div(v1*v1, (v*v).sum(dim=-1).sum(dim=-1))
            # 添加损失
            sparsity_loss += q_s.sum(dim=-1)
            sparsity_loss += k_s.sum(dim=-1)
            sparsity_loss += v_s.sum(dim=-1)

        # block.0.former.layers.0.0.fn.to_out.0.weight:torch.Size([192, 64])
        # row
        if 'fn.to_out' in name and 'weight' in name:
            # 在行上计算L2正则化
            out = param.data
            out_norm = out.norm(2, dim=-2)
            # 在列上求和
            out1 = out_norm.sum(dim=-1)
            # 求结构化稀疏度正则项
            out_s = torch.div(out1*out1, (out*out).sum(dim=-1).sum(dim=-1))
            # 添加损失
            sparsity_loss += out_s.sum(dim=-1)

        # block.0.former.layers.0.1.fn.net.0.weight:torch.Size([384, 192])
        # column
        if 'fn.net.0' in name and 'weight' in name:
            w1 = param.data
            # 在列上计算L2正则化
            w1_norm = w1.norm(2, dim=-1)
            # 在行上求和
            w11 = w1_norm.sum(dim=-1)
            # 求结构化稀疏度正则项
            w1_s = torch.div(w11*w11, (w1*w1).sum(dim=-1).sum(dim=-1))
            # 添加损失
            sparsity_loss += w1_s.sum(dim=-1)

        # block.0.former.layers.0.1.fn.net.3.weight:torch.Size([192, 384])
        # row
        if 'fn.net.3' in name and 'weight' in name:
            # 在行上计算L2正则化
            w2 = param.data
            w2_norm = w2.norm(2, dim=-2)
            # 在列上求和
            w21 = w2_norm.sum(dim=-1)
            # 求结构化稀疏度正则项
            w2_s = torch.div(w21*w21, (w2*w2).sum(dim=-1).sum(dim=-1))
            # 添加损失
            sparsity_loss += w2_s.sum(dim=-1)

    return sparsity_loss


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

            # 每100个iteration打印一次测试集准确率
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
    model_save_dir = check_point_dir + 'mobile_former_151.pt'

    model = model.to(device)

    for e in range(epochs):
        model.train()
        total_loss = 0
        for t, (x, y) in enumerate(loader_train):
            x = x.to(device=device, dtype=dtype, non_blocking=True)
            y = y.to(device=device, dtype=torch.long, non_blocking=True)

            # 原x+混x，原t，混t，原混比
            inputs, targets_a, targets_b, lam = cutmix(x, y, 1)
            # 原x+混x->原y+混y
            outputs = model(inputs)

            # 原y+混y和原t，混t求损失：lam越大，小方块越小，被识别成真图片的概率越大
            # 2
            loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam) + prune_loss(model)
            loss_value = np.array(loss.item())
            total_loss += loss_value

            # 1
            optimizer.zero_grad()
            # 3
            loss.backward()
            # 4
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # 200个iteration打印一下训练集损失
            if t % 200 == 0:
                print("Iteration:" + str(t) + ', average Loss = ' + str(loss_value))

        total_loss /= t
        losses.append(total_loss)

        acc = check_accuracy(loader_val, model, device=device)
        accs.append(np.array(acc))

        # 每个epoch记录一次测试集准确率和所有batch的平均训练损失
        print("Epoch:" + str(e) + ', Val acc = ' + str(acc) + ', average Loss = ' + str(total_loss))
        # 将每个epoch的平均损失写入文件
        with open("./tune_model/avgloss.txt", "a") as file1:
            file1.write(str(total_loss) + '\n')
        file1.close()
        # 将每个epoch的测试集准确率写入文件
        with open("./tune_model/testacc.txt", "a") as file2:
            file2.write(str(acc) + '\n')
        file2.close()

        # 如果到了保存的epoch或者是训练完成的最后一个epoch
        # if (e % save_epochs == 0 and e != 0) or e == epochs - 1 or acc >= 0.765:
        if acc > 0.7829:
            np.save(record_dir_acc, np.array(accs))
            np.save(record_dir_loss, np.array(losses))
            model.eval()
            # 保存模型参数
            torch.save(model.state_dict(), './tune_model/mobile_former_151.pth')
            # 保存模型结构
            torch.save(model, './tune_model/mobile_former_151.pt')
            # 保存jit模型
            trace_model = torch.jit.trace(model, torch.Tensor(1, 3, 224, 224).cuda())
            torch.jit.save(trace_model, './tune_model/mobile_former_jit.pt')
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
    # 50000无增强，50000有增强
    cifar_train = dset.CIFAR100('./dataset/', train=True, download=True, transform=transform)
    cifar_train_aug = dset.CIFAR100('./dataset/', train=True, download=True, transform=transform_aug)
    cifar_train += cifar_train_aug

    loader_train = DataLoader(cifar_train, batch_size=128, shuffle=True, pin_memory=True)
    print(len(cifar_train))

    # 10000无增强，10000有增强
    cifar_val = dset.CIFAR100('./dataset/', train=False, download=True, transform=transform)
    # cifar_val_aug = dset.CIFAR100('./dataset/', train=False, download=True, transform=transform_aug)
    # cifar_val += cifar_val_aug

    loader_val = DataLoader(cifar_val, batch_size=64, shuffle=True, pin_memory=True)
    print(len(cifar_val))

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
        'model': mobile_former_151(100, pre_train=True, state_dir='./bridge_ablation/mobile_former_151.pt'),
        'criterion': nn.CrossEntropyLoss(),
        # 余弦退火
        'T_mult': 2,
        'epoch': 300, 'lr': 0.0009, 'wd': 0.10,
        'check_point_dir': './tune_model/', 'save_epochs': 3,
    }
    run(**args)