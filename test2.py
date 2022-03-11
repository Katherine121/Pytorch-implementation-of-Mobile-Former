import time
import torch
from torchvision import transforms
from PIL import Image

from model_generator import mobile_former_151
from test import TestModel

if __name__ == "__main__":
    # 加载pt模型
    model = torch.load("./170/mobile_former_151.pt")
    model.eval()

    # m2f单测试成功
    # 192,2,12
    # torch.Tensor(128,12,112,112), torch.Tensor(128, 6, 192)

    # f2m单测试成功
    # 192,2,16
    # torch.Tensor(128,16,56,56), torch.Tensor(128,6,192)

    # f
    # 192
    # torch.Tensor(128,6,192)
    # json

    # m
    # 3, 12, 72, 16, None, 2, 192
    # torch.Tensor(128, 12, 112, 112), torch.Tensor(128, 6, 192)
    # need more than 1 value to unpack

    # 合起来：广播


    # 保存jit模型
    trace_model = torch.jit.trace(model, torch.rand(1,3,224,224))
    trace_model.eval()
    torch.jit.save(trace_model, './170/mobile_former_jit.pt')
    # 加载jit模型
    # trace_model = torch.jit.load('./170/mobile_former_jit.pt', map_location=torch.device('cpu'))
    # trace_model.eval()
    #
    # image_PIL = Image.open('./test_img/orange.jpg')
    # transform = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])
    # image_tensor = transform(image_PIL)
    # # 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
    # image_tensor.unsqueeze_(0)
    # print(image_tensor.shape)
    # # image_tensor = image_tensor.to(device)
    #
    # starttime = time.time()
    # out = trace_model(image_tensor)
    # endtime = time.time()
    # print(int(round((endtime - starttime) * 1000)))
    # print(out.shape)
    # # 得到预测结果，并且从大到小排序
    # _, preds = out.max(1)
    # print(preds)
