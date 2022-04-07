import torch
torch.set_printoptions(profile="full")
from model_generator import *

# python -m onnxsim rk3399_model/mobile_former_151.onnx rk3399_model/mobile_former_sim.onnx
if __name__ == '__main__':
    model = mobile_former_151(100, pre_train=True, state_dir='./acc/mobile_former_151.pth')
    model.cpu()
    model.eval()
    # for name, param in model.named_parameters():
    #     if 'token' in name:
    #         print(param)

    x = torch.Tensor(1,3,224,224)

    export_onnx_file = "./acc/mobile_former_151.onnx"  # 目的ONNX文件名
    torch.onnx.export(model,
                      x,
                      export_onnx_file,
                      verbose=True,
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=["input"],  # 输入名
                      output_names=["output"],  # 输出名
                      opset_version=11,
                      # dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                      #               "output": {0: "batch_size"}},
                      )