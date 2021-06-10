import torch
import torch.utils.data as data
from torch.autograd import Variable as V
import PIL.Image as Image

import cv2
import numpy as np
import os
import scipy.misc as misc

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask
#水平翻转
def randomHorizontalFlip(image, mask1, u=0.5):
    
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask1 = cv2.flip(mask1, 1)
    
    return image, mask1
#垂直翻转
def randomVerticleFlip(image, mask1, u=0.5):
    
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask1 = cv2.flip(mask1, 0)
    
    return image, mask1

#顺时针90
def randomRotate90_Clockwise(image, mask1, u=0.5):
   
    if np.random.random() < u:
        image=np.rot90(image)
        mask1=np.rot90(mask1)
 
    return image, mask1

#逆时针90
def randomRotate90_Clockwise(image, mask1, u=0.5):
    
    if np.random.random() < u:
        image=np.rot90(image,-1)
        mask1=np.rot90(mask1,-1)
    
    return image, mask1




#处理LITS图像
mask = "./data/LITS_train/batch2/liver_target"

liver = "./data/LITS_train/batch2/raw"

ls_liver = os.listdir(liver)
#print(ls_liver)
n = len(ls_liver)

mask_save = "./data/LITS_train/batch2/liver_target_transform"
liver_save = "./data/LITS_train/batch2/raw_transform"

for i in range(n):
    print("{}/{}".format(i+1,n))
    root_liver = os.path.join(liver,str(ls_liver[i]))
    
    root_mask = os.path.join(mask,str(ls_liver[i]))
    img = cv2.imread(root_liver)
    mask1 = cv2.imread(root_mask)


    img,mask1 = randomRotate90_Clockwise(img,mask1)
    img,mask1 = randomRotate90_Clockwise(img,mask1)
    img,mask1 = randomHorizontalFlip(img,mask1)
    img,mask1 = randomVerticleFlip(img,mask1)

    img = Image.fromarray(img)
    mask1 = Image.fromarray(mask1)

    root_liver_save = os.path.join(liver_save,str(ls_liver[i]))
    root_mask_save = os.path.join(mask_save,str(ls_liver[i]))
    img.save(root_liver_save)
    mask1.save(root_mask_save)



'''
#处理医院数据

root = "./data/tri"
all_name = os.listdir(root)
for name in all_name:
    
    liver_dir = os.path.join(root,name,"bmp")
    mask_dir = os.path.join(root,name,"mask_bw")
    ls_liver = os.listdir(liver_dir)
    ls_mask = os.listdir(mask_dir)

    length = len(ls_liver)

    
    for i in range(length):
        print("{}/{}/{}".format(name,i+1,length))
        root_liver = os.path.join(liver_dir,ls_liver[i])
        root_mask = os.path.join(mask_dir,ls_mask[i])

        img = cv2.imread(root_liver)
        mask1 = cv2.imread(root_mask)

        img,mask1 = randomRotate90_Clockwise(img,mask1)
        img,mask1 = randomRotate90_Clockwise(img,mask1)
        img,mask1 = randomHorizontalFlip(img,mask1)
        img,mask1 = randomVerticleFlip(img,mask1)
        img = Image.fromarray(img)
        mask1 = Image.fromarray(mask1)

        root_liver_save = os.path.join(liver_dir,"liver (%d).bmp"%(length+i+1))
        root_mask_save = os.path.join(mask_dir,"liver (%d).bmp"%(length+i+1))
        img.save(root_liver_save)
        mask1.save(root_mask_save)
'''
