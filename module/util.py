import os
import numpy as np
import cupy as cp
from PIL import Image


def serial_load_rawimag(level='Easy'):
    imgPath = './picture/'+str(level)+'/'
    imgname = os.listdir(imgPath)
    image_list = []
    imgname.sort()
    for img in imgname:
        print(img)
        img = Image.open(os.path.join(imgPath, img)).convert('L')
        ima = np.array(img)/255.
        image_list.append(ima)
    return(image_list)


def cuda_load_rawimag(level='Easy'):
    imgPath = './picture/'+str(level)+'/'
    imgname = os.listdir(imgPath)
    image_list = []
    imgname.sort()
    for img in imgname:
        print(img)
        img = Image.open(os.path.join(imgPath, img)).convert('L')
        ima = np.array(img)/255.
        image_list.append(cp.asarray(ima))
    return(image_list)
