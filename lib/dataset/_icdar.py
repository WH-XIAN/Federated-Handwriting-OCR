from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2
import cv2
import math
import numpy as np
import threading
import multiprocessing
from matplotlib import pyplot as plt
from contextlib import contextmanager
import pdb
import json
from PIL import Image


with open('/mnt/Data/kaiyu/OCR/CRNN_Chinese_Characters_Rec/lib/config/alphabet_cn_Breeno_new.json') as json_file:
    label_dict = json.load(json_file)

def order_pts(pts):
    rect = np.zeros((4,2), dtype = "float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def rotate(img, cords):
    cords = order_pts(np.array(cords))
    rows,cols,_ = img.shape
    points1 = np.float32(cords)
    points2 = np.float32( [[0, cols], [0,0], [rows, 0], [rows, cols]])
    matrix = cv2.getPerspectiveTransform(points1,points2)
    output = cv2.warpPerspective(img, matrix, (rows, cols))
    return output

def crop(img, cords):
    cords = order_pts(np.array(cords))
    pt1,pt2,pt3,pt4 = cords[0],cords[1],cords[2],cords[3]
    #print(pt1,pt2,pt3,pt4)
    withRect = int(math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2))  # width
    heightRect = int(math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2))
    
    rows,cols,_ = img.shape
    points1 = np.float32(cords)
    points2 = np.float32([[0,0],[heightRect,0],[heightRect,withRect],[0,withRect]])
    matrix = cv2.getPerspectiveTransform(points1,points2)
    output = cv2.warpPerspective(img, matrix, (heightRect, withRect))
    return output

def polygons_to_mask(polys, height, width):
    """
    Convert polygons to binary masks.

    Args:
        polys: a list of nx2 float array. Each array contains many (x, y) coordinates.

    Returns:
        a binary matrix of (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    assert len(polys) > 0, "Polygons are empty!"

    import pycocotools.mask as cocomask
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)

def largest_size_at_most(height, width, largest_side, max_scale):
    """
    Compute resized image size with limited max scale. 
    """
    scale = largest_side/height if height>width else largest_side/width
    scale = min(scale, max_scale)

    new_height, new_width = height * scale, width * scale
    return new_height, new_width

def aspect_preserving_resize(image, largest_side, max_scale=4.):
    """
    Resize image with perserved aspect and limited max scale.
    """
    height, width = image.shape[:2]
    new_height, new_width = largest_size_at_most(height, width, largest_side, max_scale)

    new_height = max(new_height, 8)
    new_width = max(new_width, 8)
    resized_image = cv2.resize(image, (int(new_width), int(new_height)))

    return resized_image

def padding_image(image, padding_size):
    """
    Padding arbitrary-shaped text image to square for tensorflow batch training.
    """
    height, width = image.shape[:2]
    padding_h = padding_size[0] - height
    padding_w = padding_size[1] - width

    padding_top = np.random.randint(padding_h)
    padding_left = np.random.randint(padding_w)
    padding_down = padding_h - padding_top
    padding_right = padding_w - padding_left

    padding_img = cv2.copyMakeBorder(image, padding_top, padding_down, padding_left, padding_right, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
    return padding_img, (padding_top, padding_left, height, width)

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

def rotatedPoint(R, point):
    """
    Transform polygon with affine transform matrix.
    """
    x = R[0,0]*point[0] + R[0,1]*point[1] + R[0,2]
    y = R[1,0]*point[0] + R[1,1]*point[1] + R[1,2]
    return [int(x), int(y)]

def affine_transform(image, polygon):    
    """
    Conduct same affine transform for both image and polygon for data augmentation.
    """ 
    height, width, _ = image.shape
    center_x, center_y = width/2, height/2

    angle = 0 if np.random.uniform()>0.5 else np.random.uniform(-20., 20.)
    shear_x, shear_y = (0,0) if np.random.uniform()>0.5 else (np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2))
  
    rad = math.radians(angle)
    sin, cos = math.sin(rad), math.cos(rad)   # x, y
    abs_sin, abs_cos = abs(sin), abs(cos)

    new_width = ((height * abs_sin) + (width * abs_cos))
    new_height = ((height * abs_cos) + (width * abs_sin))
    
    new_width += np.abs(shear_y*new_height)
    new_height += np.abs(shear_x*new_width)
    
    new_width = int(new_width)
    new_height = int(new_height)
    
    M = np.array([[cos, sin+shear_y,  new_width/2 - center_x + (1-cos)*center_x-(sin+shear_y)*center_y],
                          [-sin+shear_x, cos, new_height/2 - center_y + (sin-shear_x)*center_x+(1-cos)*center_y]])

    rotatedImage = cv2.warpAffine(image, M, (new_width, new_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    height, width = rotatedImage.shape[:2]
    rotatedPoints = [rotatedPoint(M, point) for point in polygon]
    mask = polygons_to_mask([np.array(rotatedPoints, np.float32)], new_height, new_width)
    x, y, w, h = cv2.boundingRect(mask)
    mask = np.expand_dims(np.float32(mask), axis=-1)
    rotatedImage = rotatedImage
    
    cropImage = rotatedImage[y:y+h, x:x+w,:]

    return cropImage


class _icdar(data.Dataset):
    def __init__(self, config, is_train=True):

        self.train = config.DATASET.TRAIN
        self.val = config.DATASET.VAL
        self.is_train = is_train
        if is_train:
            self.data = np.load(self.train, allow_pickle=True)[()]
        else:
            self.data = np.load(self.val, allow_pickle=True)[()]
        
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        # convert name:indices to name:string
        self.labels_orig = self.data['labels']
        self.img_name = self.data['filenames']
        self.cord = self.data['points']

        self.labels = []
        for i in range(len(self.labels_orig)):
            label = [label_dict[idx] for idx in self.labels_orig[i]]
            string = ''.join(label)
            self.labels.append({i: string})

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_name = self.img_name[idx]
        bbox = self.cord[idx]
        img = cv2.imread(img_name)
        crop_img = crop(img, bbox)
        img_h, img_w, _ = crop_img.shape
        
        if img_h>2*img_w:
            crop_img = rotate(crop_img,[[0,0], [img_w, 0], [img_w, img_h], [0, img_h]])

        crop_img = Image.fromarray(crop_img).convert('L')
        img1 = img_preprocessing(crop_img)

        img = np.array(img1)

        img = np.reshape(img, (self.inp_h, self.inp_w, 1))
        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img, idx
        
if __name__ == '__main__':
    datafile = np.load('/mnt/Data/kaiyu/OCR/CRNN_Chinese_Characters_Rec/lib/dataset/txt/icdar_train.npy', allow_pickle=True)[()]
    for i in range(10):
        img = cv2.imread(datafile['filenames'][i])
        label = datafile['labels'][i]
        print(label)
        label = [label_dict[idx] for idx in label]
        string = ''.join(label)
        print(string)
        bbox = datafile['points'][i]

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crop_img = crop(img, bbox)
        img_h, img_w, _ = crop_img.shape
        if img_h>2*img_w:
            crop_img = rotate(crop_img,[[0,0], [img_w, 0], [img_w, img_h], [0, img_h]])

        crop_img = Image.fromarray(crop_img).convert('L')
        img1 = img_preprocessing(crop_img)
        img = np.array(img1)
        cv2.imwrite('images/'+str(i)+'.jpg', img)



    