from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2
import json
from PIL import Image
import random

def img_preprocessing(img):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)

    if width<320:
        img = img.resize([width, 32], Image.ANTIALIAS)
        bg = Image.new('L', (320, 32))
        bg.paste(img)
    else:
        bg = img.resize([320, 32], Image.ANTIALIAS)
    return bg

def img_encode(img):
    _, im = cv2.imencode('.jpg', img, [cv2.IMWRITE_WEBP_QUALITY, 90])
    im = cv2.imdecode(im, cv2.IMREAD_COLOR)
    return img

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

class _OWN(data.Dataset):
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
        self.quality = random.randint(0,1)

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_name = list(self.labels[idx].keys())[0]
        img1 = Image.open(img_name).convert('L')
        width, height = img1.size[0], img1.size[1]
        if height > 1.2*width:
            img1 = np.array(img1)
            img_h, img_w = img1.shape
            img1 = rotate(img1,[[0,0], [img_w, 0], [img_w, img_h], [0, img_h]])
            img1 = Image.fromarray(img1)
        img1 = img_preprocessing(img1)
        img = np.array(img1)

        if self.quality:
            img = img_encode(img)

        img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img, idx


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





