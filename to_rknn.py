# 不能使用torch1.10.0导出，会出问题
import cv2
import numpy as np
from rknn.api import RKNN

if __name__ == '__main__':
    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.82, 58.82, 58.82]], reorder_channel='0 1 2')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model='./rk3399_model/mobile_former_151.onnx')
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
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

    # Set inputs
    img = cv2.imread('./test_img/orange.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    x = outputs[0]
    output = np.exp(x) / np.sum(np.exp(x))
    outputs = [output]
    print(outputs)
    print('done')

    rknn.release()