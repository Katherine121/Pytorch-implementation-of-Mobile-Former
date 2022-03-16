from model_generator import *

if __name__ == '__main__':
    model = mobile_former_151(100, pre_train=True, state_dir="./saved_model/mobile_former_151.pt")
    model.cpu()
    model.eval()

    # 使用onnxoptimizer优化模型
    # model = onnxoptimizer.optimize(model)

    x = torch.Tensor(1,3,224,224)

    export_onnx_file = "./rk3399_model/mobile_former_151.onnx"  # 目的ONNX文件名
    torch.onnx.export(model,
                      x,
                      export_onnx_file,
                      verbose=True,
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=["input"],  # 输入名
                      output_names=["output"],  # 输出名
                      opset_version=11,
                      )