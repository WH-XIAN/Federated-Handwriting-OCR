from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2
import json
from PIL import Image

def image_chunking(image):
    # 这里我要写死chunks的个数 暂时先限定 3 * 320 ==> 960 thanks ben help me figuring this out
    image = np.array(image)
    h, wid = image.shape
    num_chunks = 3
    image_chunks = [np.zeros((32, 320), dtype=np.float32) for i in range(num_chunks)]
    valid_chunks = 1
    if wid == 320:
        image_chunks[0] = image
        return image_chunks, 1 
    import math
    valid_num_chunks = math.ceil(wid / 256)
    padded_img = np.zeros((32, 32 + wid ), dtype=np.float32)
    padded_img[:, 32:] = image[:, :]
    padded_wid = padded_img.shape[1]
    
    cur = 0
    print('image length is {0}, num of chunks is {1} '.format(wid, valid_num_chunks))
    for idx, chunk in enumerate(image_chunks):
        if idx >= valid_num_chunks : break
        
        if cur + 320 <= padded_wid:
            chunk[:, :] = padded_img[:, cur: cur + 320] 
        else:
            chunk[:, : padded_wid - cur ] = padded_img[:, cur: padded_wid]
        cur += 256

    # import uuid
    # reid = str(uuid.uuid4())
    # root = '/mnt/Data02/junxi/CRNN_Chinese_Characters_Rec/chunking_test'
    # for idx, chunk in enumerate(image_chunks):
    #     cv2.imwrite(root + "/{0}_{1}.jpg".format(reid, idx), chunk)
    # cv2.imwrite(root + "/{0}_{1}.jpg".format(reid, 100), image)
    return image_chunks, valid_num_chunks

def norm_and_reshape(bg, inp_h, inp_w, mean, std):
    
    img = np.array(bg)

    img = np.reshape(img, (inp_h, inp_w, 1))

    img = img.astype(np.float32)
    img = (img/255. - mean) / std
    img = img.transpose([2, 0, 1])

    return img

def fixed_img_preprocessing(img, inp_h, inp_w, mean, std):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)
    
    if width < inp_w:
        resized_img = img.resize([width, 32], Image.ANTIALIAS)
        bg = Image.new('L', (inp_w, 32))
        bg.paste(resized_img)
       
    else:
        bg = img.resize([inp_w, 32], Image.ANTIALIAS)
        
    processed_image = norm_and_reshape(bg, inp_h, inp_w, mean, std)
        
    return processed_image 

def img_preprocessing(img, inp_h, inp_w, mean, std):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)
    
    if width < 320:
        resized_img = img.resize([width, 32], Image.ANTIALIAS)
        bg = Image.new('L', (320, 32))
        bg.paste(resized_img)
       
    else:
        if width > 768:
            bg = img.resize([768, 32], Image.ANTIALIAS)
        else:
            bg = img.resize([width, 32], Image.ANTIALIAS)
        
    image_chunks, valid_num_chunks = image_chunking(bg)
    image_chunks = norm_and_reshape(image_chunks, inp_h, inp_w, mean, std)
        
    return image_chunks, valid_num_chunks

def rotate(img, cords):
    cords = order_pts(np.array(cords))
    rows,cols = img.shape
    points1 = np.float32(cords)
    points2 = np.float32( [[0, cols], [0,0], [rows, 0], [rows, cols]])
    matrix = cv2.getPerspectiveTransform(points1,points2)
    output = cv2.warpPerspective(img, matrix, (rows, cols))
    return output

def order_pts(pts):
    rect = np.zeros((4,2), dtype = "float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

class _Latin(data.Dataset):
    def __init__(self, config, is_train=True):

        self.root = config.DATASET.ROOT
        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        txt_file = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']

        # convert name:indices to name:string
        with open(txt_file) as file:
            self.labels = json.load(file)
        self.labels = [{key: self.labels[key]} for key in self.labels]

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        img_name = list(self.labels[idx].keys())[0]
        label = list(self.labels[idx].values())[0]
        img1 = Image.open(img_name).convert('L')
        width, height = img1.size[0], img1.size[1]
        if height > 1.5*width:
            img1 = np.array(img1)
            img_h, img_w = img1.shape
            img1 = rotate(img1,[[0,0], [img_w, 0], [img_w, img_h], [0, img_h]])
            img1 = Image.fromarray(img1)
        # img1 = img_preprocessing(img1)
        processed_image = fixed_img_preprocessing(img1, self.inp_h, self.inp_w, self.mean, self.std)
        
        return processed_image, label, idx


if __name__ == '__main__':
    img_name = '/mnt/Data02/kaiyu/Data/OCR/Breeno/JpKr/8928.jpg'
    img1 = Image.open(img_name).convert('L')
    width, height = img1.size[0], img1.size[1]
    if height > 1.2*width:
        img1 = np.array(img1)
        img_h, img_w = img1.shape
        img1 = rotate(img1,[[0,0], [img_w, 0], [img_w, img_h], [0, img_h]])
        img1 = Image.fromarray(img1)
    img1 = img_preprocessing(img1)
    img1.save('/mnt/Data/kaiyu/OCR/CRNN_Chinese_Characters_Rec/lib/dataset/images/2.jpg')





