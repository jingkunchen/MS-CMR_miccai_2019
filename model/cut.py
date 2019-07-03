#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import SimpleITK as sitk
import scipy.misc
from skimage.transform import resize
# from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage

import cv2
import time
from decimal import Decimal
import skimage.io as io

data_dir = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0t2lge/'
thresh = 1
rows = 256
cols = 256
xmin = 1 
xmax = 1
ymin = 1
ymax = 1
xlenmin = 1
ylenmin = 1

img_count = 0
def show_img(data):
        io.imshow(data[:,:], cmap = 'gray')
        io.show()
def show_img_all(data):
    for i in range(data.shape[0]):
        io.imshow(data[i, :, :], cmap='gray')
        io.show()


# label transform, 500-->1, 200-->2, 600-->3
###### LGE
LGE_data_1ch = []
LGE_gt_1ch = []
img_dir = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/lge_images/'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
gt_dir_1 = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/lgegt/'
lge_list = []
for pp in range(1, 6):

    data_name = data_dir + 'patient' + str(pp) + '_LGE.nii.gz'
    gt_name = gt_dir_1 + 'patient' + str(pp) + '_LGE_manual.nii.gz'
    img = sitk.ReadImage(os.path.join(gt_name))
    data_array = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(data_name)))
    gt_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_name)))
    img_count +=gt_array.shape[0]
    print(np.shape(data_array))

    x = []
    y = []
    print("idx:", pp)
    for image in gt_array:
        for i in range(np.shape(gt_array)[1]):
            for j in range(np.shape(gt_array)[2]):
                if image[i][j] != 0:
                    if i <30 or j<30:
                        print("label_error:", pp,i,j,image[i][j])
                    else:
                        x.append(i)
                        y.append(j)
    print(min(x),max(x),max(x)-min(x),round(min(x)/np.shape(gt_array)[1],2), round(max(x)/np.shape(gt_array)[1],2))
    print(min(y),max(y),max(y)-min(y),round(min(y)/np.shape(gt_array)[1],2), round(max(y)/np.shape(gt_array)[1],2))

    mask = np.zeros(np.shape(data_array), dtype='float32')
    mask[data_array >= thresh] = 1
    mask[data_array < thresh] = 0
    for iii in range(np.shape(data_array)[0]):
        mask[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(
            mask[iii, :, :])  #fill the holes inside br
    data_array = data_array - np.mean(data_array[mask == 1])
    data_array /= np.std(data_array[mask == 1])
    rows_o = np.shape(data_array)[1]
    cols_o = np.shape(data_array)[2]

    data_array_ = data_array[:,
                             int((rows_o - rows) /
                                 2):int((rows_o - rows) / 2) + rows,
                             int((cols_o - cols) /
                                 2):int((cols_o - cols) / 2) + cols]
    gt_array_ = gt_array[:,
                         int((rows_o - rows) /
                             2):int((rows_o - rows) / 2) + rows,
                         int((cols_o - cols) / 2):int((cols_o - cols) / 2) +
                         cols]
    mask = mask[:,
                int((rows_o - rows) / 2):int((rows_o - rows) / 2) + rows,
                int((cols_o - cols) / 2):int((cols_o - cols) / 2) + cols]

    LGE_data_1ch.extend(np.float32(data_array_))
    LGE_gt_1ch.extend(np.float32(gt_array_))

LGE_data_1ch = np.asarray(LGE_data_1ch)
LGE_gt_1ch = np.asarray(LGE_gt_1ch)
LGE_gt_1ch[LGE_gt_1ch == 500] = 1
LGE_gt_1ch[LGE_gt_1ch == 200] = 2
LGE_gt_1ch[LGE_gt_1ch == 600] = 3

# np.save('LGE_data_1ch.npy', LGE_data_1ch)
# np.save('LGE_gt_1ch.npy', LGE_gt_1ch)

T2_data_1ch = []
T2_gt_1ch = []
T2_shape = []
img_dir = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/t2_images/'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
gt_dir_1 = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/t2gt/'

for pp in range(1, 36):
    data_name = data_dir + 'patient' + str(pp) + '_T2.nii.gz'
    gt_name = gt_dir_1 + 'patient' + str(pp) + '_T2_manual.nii.gz'
    
    data_array = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(data_name)))
    gt_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_name)))
    
    data_array = np.nan_to_num(data_array, copy=True)
    gt_array = np.nan_to_num(gt_array, copy=True)
    
    img_count +=gt_array.shape[0]

    x = []
    y = []
    print("idx:", pp)
    new_gt_list = []
    for image in gt_array:
        if image.shape[0] < rows:
            image = np.asarray(image)
            image1 = image.copy()
            image2 = image.copy()
            image[image == 500] = 1
            image[image == 200] = 0
            image[image == 600] = 0
            image1[image1 == 500] = 0
            image1[image1 == 200] = 1
            image1[image1 == 600] = 0
            image2[image2 == 500] = 0
            image2[image2 == 200] = 0
            image2[image2 == 600] = 1
            
            image = resize(image,(rows,cols), preserve_range =True)
            image1 = resize(image1,(rows,cols), preserve_range =True)
            image2 = resize(image2,(rows,cols), preserve_range =True)
            image = np.around(image)
            image1 = np.around(image1)
            image2 = np.around(image2)
            image = image.astype(np.int32)
            image1 = image1.astype(np.int32)
            image2 = image2.astype(np.int32)

            
            image[image == 1] = 1
            image1[image1 == 1] = 2
            image2[image2 == 1] = 3
            image = image +image1 +image2
            [rows, cols] = image.shape
            for i in range(rows):
                for j in range(cols):
                    if(image[i, j] >3) :
                        print("--------error----------:", pp)
            image[image == 1] = 500
            image[image == 2] = 200
            image[image == 3] = 600
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] != 0:
                    if j < 40 or i < 40:
                        print("label error:", pp,i,j,image[i][j])
                        image[0:75, 0:50] = 0
                        
                    else:
                        x.append(i)
                        y.append(j)
        new_gt_list.append(image)            
    gt_array=np.array(new_gt_list)

    new_data_list = []
    if data_array.shape[1] < rows:
        print("idx:", pp)
        for image in data_array:
            image = np.asarray(image)
            image = resize(image, (rows, cols), preserve_range =True)
            image = np.around(image)
            image = image.astype(np.int32)
            new_data_list.append(image)
        

        data_array=np.array(new_data_list)

    print("new_array:",gt_array.shape)
    
    print(min(x), max(x),
          max(x) - min(x), round(min(x) / np.shape(gt_array)[1], 2),
          round(max(x) / np.shape(gt_array)[1], 2))
    print(min(y), max(y),
          max(y) - min(y), round(min(y) / np.shape(gt_array)[1], 2),
          round(max(y) / np.shape(gt_array)[1], 2))
    if(round(min(x)/np.shape(gt_array)[1],2) < 0.2 or round(min(y)/np.shape(gt_array)[1],2)<0.2):
        print("errorerrorerrorerrorerrorerror")
    # 256 288 240 240 224 
    mask = np.zeros(np.shape(data_array), dtype='float32')
    mask[data_array >= thresh] = 1
    mask[data_array < thresh] = 0

    for iii in range(np.shape(data_array)[0]):
        mask[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(
            mask[iii, :, :])  #fill the holes inside br

    data_array = data_array - np.mean(data_array[mask == 1])
    data_array /= np.std(data_array[mask == 1])
    rows_o = np.shape(data_array)[1]
    cols_o = np.shape(data_array)[2]

    data_array_ = data_array[:,
                             int((rows_o - rows) /
                                 2):int((rows_o - rows) / 2) + rows,
                             int((cols_o - cols) /
                                 2):int((cols_o - cols) / 2) + cols]
    gt_array_ = gt_array[:,
                         int((rows_o - rows) /
                             2):int((rows_o - rows) / 2) + rows,
                         int((cols_o - cols) / 2):int((cols_o - cols) / 2) +
                         cols]

    print("np.max(data_array_):",np.max(data_array_))
    T2_data_1ch.extend(np.float32(data_array_))
    T2_gt_1ch.extend(np.float32(gt_array_))




T2_data_1ch = np.asarray(T2_data_1ch)
T2_gt_1ch = np.asarray(T2_gt_1ch)
T2_gt_1ch[T2_gt_1ch == 500] = 1
T2_gt_1ch[T2_gt_1ch == 200] = 2
T2_gt_1ch[T2_gt_1ch == 600] = 3
# np.save('T2_data_1ch.npy', T2_data_1ch)
# np.save('T2_gt_1ch.npy', T2_gt_1ch)

#######C0
#
C0_data_1ch = []
C0_gt_1ch = []
img_dir = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0_images/'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
gt_dir_1 = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0gt/'

for pp in range(1, 36):
    data_name = data_dir + 'patient' + str(pp) + '_C0.nii.gz'
    gt_name = gt_dir_1 + 'patient' + str(pp) + '_C0_manual.nii.gz'
    data_array = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(data_name)))
    gt_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_name)))
    print(np.shape(data_array))
    img_count +=gt_array.shape[0]



    x = []
    y = []

    new_gt_list = []
    for image in gt_array:
        if image.shape[0] < rows:
            image = np.asarray(image)
            image1 = image.copy()
            image2 = image.copy()
            image[image == 500] = 1
            image[image == 200] = 0
            image[image == 600] = 0
            image1[image1 == 500] = 0
            image1[image1 == 200] = 1
            image1[image1 == 600] = 0
            image2[image2 == 500] = 0
            image2[image2 == 200] = 0
            image2[image2 == 600] = 1
            
            image = resize(image,(rows,cols), preserve_range =True)
            image1 = resize(image1,(rows,cols), preserve_range =True)
            image2 = resize(image2,(rows,cols), preserve_range =True)
            image = np.around(image)
            image1 = np.around(image1)
            image2 = np.around(image2)
            image = image.astype(np.int32)
            image1 = image1.astype(np.int32)
            image2 = image2.astype(np.int32)

            
            image[image == 1] = 1
            image1[image1 == 1] = 2
            image2[image2 == 1] = 3
            image = image +image1 +image2
            [rows, cols] = image.shape
            for i in range(rows):
                for j in range(cols):
                    if(image[i, j] >3) :
                        print("--------error----------:", pp)
            image[image == 1] = 500
            image[image == 2] = 200
            image[image == 3] = 600
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] != 0:
                    if j < 40 or i < 40:
                        print("label error:", pp,i,j,image[i][j])
                        image[0:75, 0:50] = 0
                        
                    else:
                        x.append(i)
                        y.append(j)
        new_gt_list.append(image)            
    gt_array=np.array(new_gt_list)

    new_data_list = []
    if data_array.shape[1] < rows:
        print("idx:", pp)
        for image in data_array:
            image = np.asarray(image)
            image = resize(image, (rows, cols), preserve_range =True)
            image = np.around(image)
            image = image.astype(np.int32)
            new_data_list.append(image)
        

        data_array=np.array(new_data_list)

    print("new_array:",gt_array.shape)


 
    print("idx:", pp)
    print(min(x), max(x),
          max(x) - min(x), round(min(x) / np.shape(gt_array)[1], 2),
          round(max(x) / np.shape(gt_array)[1], 2))
    print(min(y), max(y),
          max(y) - min(y), round(min(y) / np.shape(gt_array)[1], 2),
          round(max(y) / np.shape(gt_array)[1], 2))
    
    # 320 288 240 256 224
  
    mask = np.zeros(np.shape(data_array), dtype='float32')
    mask[data_array >= thresh] = 1
    mask[data_array < thresh] = 0

    for iii in range(np.shape(data_array)[0]):
        mask[iii, :, :] = scipy.ndimage.morphology.binary_fill_holes(
            mask[iii, :, :])  #fill the holes inside br
    data_array = data_array - np.mean(data_array[mask == 1])
    data_array /= np.std(data_array[mask == 1])
    rows_o = np.shape(data_array)[1]
    cols_o = np.shape(data_array)[2]

    data_array_ = data_array[:,
                             int((rows_o - rows) /
                                 2):int((rows_o - rows) / 2) + rows,
                             int((cols_o - cols) /
                                 2):int((cols_o - cols) / 2) + cols]
    gt_array_ = gt_array[:,
                         int((rows_o - rows) /
                             2):int((rows_o - rows) / 2) + rows,
                         int((cols_o - cols) / 2):int((cols_o - cols) / 2) +
                         cols]
  
    C0_data_1ch.extend(np.float32(data_array_))
    C0_gt_1ch.extend(np.float32(gt_array_))

   
C0_data_1ch = np.asarray(C0_data_1ch)
C0_gt_1ch = np.asarray(C0_gt_1ch)
C0_gt_1ch[C0_gt_1ch == 500] = 1
C0_gt_1ch[C0_gt_1ch == 200] = 2
C0_gt_1ch[C0_gt_1ch == 600] = 3
# np.save('C0_data_1ch.npy', C0_data_1ch)
# np.save('C0_gt_1ch.npy', C0_gt_1ch)
print("LGE_data_1ch:",LGE_data_1ch.shape)
print("C0_data_1ch:",C0_data_1ch.shape)
print("T2_data_1ch:",T2_data_1ch.shape)

new_data_array = np.concatenate((LGE_data_1ch, C0_data_1ch), axis=0)
new_data_array = np.concatenate((new_data_array, LGE_data_1ch), axis=0)
new_data_array = np.concatenate((new_data_array, T2_data_1ch), axis=0)
new_gt_array = np.concatenate((LGE_gt_1ch, C0_gt_1ch), axis=0)
new_gt_array = np.concatenate((new_gt_array, LGE_gt_1ch), axis=0)
new_gt_array = np.concatenate((new_gt_array, T2_gt_1ch), axis=0)
train_imgs_new  = new_data_array.copy()
train_masks_new  = new_gt_array.copy()
count_i = 0
count = 0
count_list = []
for image in new_gt_array:
    max_1 = np.max(image)
    
    if max_1 < 0:
        delete_number = count_i - count
        train_imgs_new = np.delete(train_imgs_new, delete_number, axis=0)
        train_masks_new = np.delete(train_masks_new, delete_number, axis=0)

        count += 1
        print("empty:",count, count_i)
        
    count_i +=1
new_data_array = train_imgs_new
new_gt_array = train_masks_new
print("new_gt_array:", new_gt_array.shape)
print("new_gt_array:", new_gt_array.shape)


output_path = "/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/del/all_data_or_256_256.nii.gz"
sitk.WriteImage(sitk.GetImageFromArray(new_data_array),output_path)
output_path = "/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/del/all_gt_or_256_256.nii.gz"
sitk.WriteImage(sitk.GetImageFromArray(new_gt_array),output_path)


np.save('/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/del/all_data_or_256_256.npy', new_data_array[:, :, :, np.newaxis])
np.save('/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/del/all_gt_or_256_256.npy', new_gt_array[:, :, :, np.newaxis])
print("img_count:",img_count)
print("new_gt_array:",new_gt_array.shape)
# print(xmin,xmax,ymin,ymax, xlenmin, ylenmin)