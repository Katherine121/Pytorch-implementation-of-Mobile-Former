# 不能使用torch1.10.0导出，会出问题
import tensorflow
import torch
import torchvision
from rknn.api import RKNN
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0.485, 0.456, 0.406]], std_values=[[0.229, 0.224, 0.225]], reorder_channel='0 1 2')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model='./rk3399_model/mobile_former_151.onnx')
    # ret = rknn.load_rknn('./rk3399_model/mobile_former_151.rknn')
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    # ret = rknn.build(do_quantization=True, dataset='dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./rk3399_model/mobile_former_151.rknn')
    if ret != 0:
        print('Export model.rknn failed!')
        exit(ret)
    print('done')

    ret = rknn.load_rknn('./rk3399_model/mobile_former_151.rknn')

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime(target='rk1808', device_id='bde4c665cdab7689')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # # get npu version
    # print("get npu version")
    # print(rknn.get_sdk_version())

    # Inference
    print('--> Running model')
    num_correct = 0
    num_samples = 0
    total_correct = 0
    total_samples = 0

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    cifar_val = torchvision.datasets.CIFAR100('./dataset/', train=False, download=True, transform=transform)

    loader_val = DataLoader(cifar_val, batch_size=1, shuffle=True)
    print(len(cifar_val))

    t = 0
    with torch.no_grad():
        for x, y in loader_val:
            outputs = rknn.inference(inputs=[x.numpy()], data_format='nchw')

            # 计算预测标签索引
            data = outputs[0][0]
            data = data.tolist()
            prob = max(data)
            idx = data.index(prob)

            num_correct = (idx == y)
            total_correct += num_correct
            print(t)
            t += 1

        acc = float(total_correct) / (t + 1)
    print(acc)

    print('done')

    rknn.release()
