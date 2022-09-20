import onnx
import torch

import sys, os
import config.alphabets_new as alphabets_raw # 
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable
from torch.nn.functional import softmax
from hw_mobile_net import CamNet

len_classes = len(alphabets_raw.alphabet_cn)

path_512 = '/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/output/Hw_Ch&En/hw_512_full_data/2022-08-02-00-23/checkpoints/checkpoint_36_acc_0.7442.pth'
# path_512 = '/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/output/Hw_Ch&En/hw_512_full_data/2022-07-15-16-43/checkpoints/checkpoint_20_acc_0.6705.pth' 
# '/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/output/Hw_Ch&En/hw_512_full_data/2022-07-06-17-41/checkpoints/checkpoint_34_acc_0.6676.pth'
#'/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/output/Hw_Ch&En/hw_512_full_data/2022-07-06-17-42/checkpoints/checkpoint_5_acc_0.6462.pth'
#'/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/output/Hw_Ch&En/hw_512_full_data/2022-06-20-17-57/checkpoints/checkpoint_40_acc_0.6402.pth' # 512

# path = '/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/output/Hw_Ch&En/hw_320_full_data/2022-04-19-17-40/checkpoints/checkpoint_63_acc_0.6064.pth' # 
path_320 = '/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/output/Hw_Ch&En/hw_320_full_data/2022-06-20-18-14/checkpoints/checkpoint_54_acc_0.6054.pth' # 320

test_image_length = 512

path = path_512 if test_image_length == 512 else path_320

pytorch_model = CamNet(len_classes, -1, img_w=test_image_length)
pytorch_model.load_weights(path, use_gpu=False)

from PIL import Image
img = Image.open('/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/lib/models/test_img.png')
width, height = img.size[0], img.size[1]
scale = height * 1.0 / 32
width = int(width / scale)

if width < test_image_length:
    resized_img = img.resize([width, 32], Image.ANTIALIAS)
    bg = Image.new('L', (test_image_length, 32))
    bg.paste(resized_img)
    
else:
    bg = img.resize([test_image_length, 32], Image.ANTIALIAS)

image_data = (np.array(bg)/255.0 - 0.588)/0.193
image_data = image_data.astype(np.float32)
image_data = torch.from_numpy(image_data)
image_data = image_data.view(1,1, 32, test_image_length)


example_input = image_data # exmample for the forward pass input 
pytorch_model = pytorch_model
ONNX_PATH="./hw_rec_model_pad_0809.onnx"

torch.onnx.export(
    model=pytorch_model,
    args=example_input, 
    f=ONNX_PATH, # where should it be saved
    verbose=False,
    export_params=True,
    do_constant_folding=False,  # fold constant values for optimization
    # do_constant_folding=True,   # fold constant values for optimization
    input_names=['input'],
    output_names=['output'],
    opset_version=11,
)
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)