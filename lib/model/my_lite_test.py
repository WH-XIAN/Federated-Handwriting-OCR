import os, sys, torch
# from time import time
import time
import tensorflow as tf
import numpy as np
import editdistance
sys.path.append(os.path.abspath('/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/lib/'))
import config.alphabets_new as alphabets_raw # 
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable
from torch.nn.functional import softmax

def strLabelConverter(res,alphabet):
    N = len(res)
    raw = []
    for i in range(N):
        if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
            raw.append(alphabet[res[i] - 1])
    return ''.join(raw)


from PIL import Image

def img_preprocess(path, img_wid=320):
    img = Image.open(path)
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)

    if width < img_wid:
        resized_img = img.resize([width, 32], Image.ANTIALIAS)
        bg = Image.new('L', (img_wid, 32))
        bg.paste(resized_img)
        
    else:
        bg = img.resize([img_wid, 32], Image.ANTIALIAS)

    image_data = (np.array(bg)/255.0- 0.588)/0.193
    image_data = image_data.astype(np.float32)
    image_data = np.expand_dims(image_data, axis=0)
    example_input = np.expand_dims(image_data, axis=0)
    
    return example_input

TFLITE_PATH_512 = './hw_rec_model_pad_0809.tflite' # "./hw_rec_model_pad.tflite"
test_width = 512
# TFLITE_PATH = '/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/lib/models/hw_rec_model_wid320_updated.tflite' if test_width == 320 else TFLITE_PATH_512
TFLITE_PATH = TFLITE_PATH_512
print('TFLITE_PATH ===>', TFLITE_PATH)
print(f"Using tensorflow {tf.__version__}") # make sure it's the nightly build
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
interpreter = tf.compat.v1.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def inference_builder(example_input):
    interpreter.set_tensor(input_details[0]['index'], example_input)
    interpreter.invoke()
    ret = interpreter.get_tensor(output_details[0]['index'])
    print(interpreter.get_tensor(output_details[0]['index']).shape) # printing the result

    # results to string
    preds = torch.from_numpy(ret)
    values, prob = softmax(preds, dim=-1).max(2)
    preds_idx = (prob > 0).nonzero()
    sent_prob = values[preds_idx[:,0], preds_idx[:, 1]].detach()
    # get preds
    _, preds    = preds.max(2)
    preds       = preds.transpose(1, 0).contiguous().view(-1)
    raw         = strLabelConverter(preds, alphabets_raw.alphabet_cn)
    print(raw, sent_prob)
    
    return raw, sent_prob

import json

tr_ret = {}
test_path = '/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/2000_hw_test.json' # /mnt/Data/junxi/ocr_data/chinese/500_rand_dataset_for_com.json
print('test path ===>', test_path)
def test():
    # with open('/mnt/Data/junxi/ocr_data/chinese/test_data_for_model_compare.json') as hw_f:
    #     hw_data = json.load(hw_f)
    with open(test_path) as hw_f:
        hw_data = json.load(hw_f)
    avg_distance = 0
    cnt = 0
    
    for path, gt in hw_data.items():
        try:
            t0 = time.time()
            image_data = img_preprocess(path, test_width)
            preds, probs = inference_builder(image_data)
            print('inferenc time ===>',time.time() - t0 )
            print(preds, probs, gt)
            res = editdistance.eval(preds, gt)
            print('editdistance is: ', res)
            avg_distance += (res / len(gt))
            print('weighted dis is: ', res / len(gt))
            cnt += 1
            tr_ret[path] = res
        except Exception as e:
            print("error information ===> ", e)
            continue
    
    # with open('/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/hw_test_results.json', 'w') as f:
    #     json.dump(tr_ret, f) 
    
    print('test results ====> ', avg_distance/cnt)

if __name__ == "__main__":
    test()
    
    
