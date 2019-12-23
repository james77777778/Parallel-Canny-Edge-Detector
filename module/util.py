import os
import numpy as np
import cupy as cp
from PIL import Image
from skimage.color import rgb2gray


def serial_load_rawimag(level='Easy'):
    imgPath = './picture/'+str(level)+'/'
    imgname = os.listdir(imgPath)
    image_list = []
    imgname.sort()
    for img in imgname:
        print(img)
        img = Image.open(os.path.join(imgPath, img))
        img.load()
        ima = np.asarray(img, dtype="int32")
        image_list.append(rgb2gray(ima))
    return(image_list)


def cuda_load_rawimag(level='Easy'):
    imgPath = './picture/'+str(level)+'/'
    imgname = os.listdir(imgPath)
    image_list = []
    imgname.sort()
    for img in imgname:
        print(img)
        img = Image.open(os.path.join(imgPath, img))
        img.load()
        ima = np.asarray(img, dtype="int32")
        image_list.append(cp.asarray(rgb2gray(ima)))
    return(image_list)
