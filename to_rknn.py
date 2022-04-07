# 不能使用torch1.10.0导出，会出问题
import tensorflow
import torch
import torchvision
from PIL import Image
from rknn.api import RKNN
from torchvision import transforms

torch.set_printoptions(profile="full")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> Config model')
    rknn.config(reorder_channel='0 1 2')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model='./acc/mobile_former_151.onnx')
    # ret = rknn.load_rknn('./dist_model/mobile_former_151.rknn')
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
    ret = rknn.export_rknn('./acc/mobile_former_151.rknn')
    if ret != 0:
        print('Export model.rknn failed!')
        exit(ret)
    print('done')

    ret = rknn.load_rknn('./acc/mobile_former_151.rknn')

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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image_PIL = Image.open('./test_img/orange.jpg')
    image_tensor = transform(image_PIL)
    image_tensor.unsqueeze_(0)

    with torch.no_grad():
        outputs = rknn.inference(inputs=[to_numpy(image_tensor)], data_format='nchw')

        # 计算预测标签索引
        data = outputs[0][0]
        print(data)
        data = data.tolist()
        prob = max(data)
        idx = data.index(prob)
        print(idx)

    image_PIL = Image.open('./test_img/boy.jpg')
    image_tensor = transform(image_PIL)
    image_tensor.unsqueeze_(0)

    with torch.no_grad():
        outputs = rknn.inference(inputs=[to_numpy(image_tensor)], data_format='nchw')

        # 计算预测标签索引
        data = outputs[0][0]
        print(data)
        data = data.tolist()
        prob = max(data)
        idx = data.index(prob)
        print(idx)

    image_PIL = Image.open('./test_img/space_shuttle_224.jpg')
    image_tensor = transform(image_PIL)
    image_tensor.unsqueeze_(0)

    with torch.no_grad():
        outputs = rknn.inference(inputs=[to_numpy(image_tensor)], data_format='nchw')

        # 计算预测标签索引
        data = outputs[0][0]
        print(data)
        data = data.tolist()
        prob = max(data)
        idx = data.index(prob)
        print(idx)

    print('done')
    rknn.release()
