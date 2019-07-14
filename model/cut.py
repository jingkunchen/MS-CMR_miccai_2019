#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import SimpleITK as sitk
import scipy.misc
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage
import cv2
import time
from decimal import Decimal
import skimage.io as io

data_dir = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0t2lge/'
thresh = 1
rows = 224
cols = 224
xmin = 1 
xmax = 1
ymin = 1
ymax = 1
xlenmin = 1
ylenmin = 1

img_count = 0
def show_img(data):
    for i in range(data.shape[0]):
        io.imshow(data[i, :, :], cmap='gray')
        # io.imshow(data[:,:], cmap = 'gray')
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
for pp in range(1, 4):

    data_name = data_dir + 'patient' + str(pp) + '_LGE.nii.gz'
    gt_name = gt_dir_1 + 'patient' + str(pp) + '_LGE_manual.nii.gz'
    img = sitk.ReadImage(os.path.join(gt_name))
    data_array = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(data_name)))
    gt_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_name)))
    img_count +=gt_array.shape[0]
    
    print(np.shape(data_array))


    # new_data_array = 0
    # count = 0
    # for image in data_array:
    #     new_image = resize(image, (480,480), anti_aliasing=False)
    #     print()
    #     if count == 0:
    #             new_data_array = new_image[np.newaxis,:,:]
    #     else:
    #             new_data_array = np.concatenate((new_data_array, new_image[np.newaxis,:,:]), axis=0)

    #     count += 1
    # data_array = new_data_array
    # new_gt_array = 0
    # count = 0
    # for gt in gt_array:

    #     new_gt = resize(gt, (480,480), anti_aliasing=False)
    #     for i in range(480):
    #             for j in range(480):
    #                     if new_gt[i][j] > 0.4:
    #                             print(new_gt[i][j])

    #     if count == 0:
    #             new_gt_array = new_gt[np.newaxis,:,:]
    #     else:
    #             new_gt_array = np.concatenate((new_gt_array, new_gt[np.newaxis,:,:]), axis=0)

    #     count += 1
    # gt_array = new_gt_array

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
    # if xmin > round(min(x)/np.shape(gt_array)[1],2):
    #     xmin = round(min(x)/np.shape(gt_array)[1],2)
    # if xmax > round(max(x)/np.shape(gt_array)[1],2):
    #     xmax = round(max(x)/np.shape(gt_array)[1],2)
    # if ymin > round(min(y)/np.shape(gt_array)[1],2):
    #     ymin = round(min(y)/np.shape(gt_array)[1],2)
    # if ymax > round(max(y)/np.shape(gt_array)[1],2):
    #     ymax = round(max(y)/np.shape(gt_array)[1],2)
    # if xlenmin > round(max(x)/np.shape(gt_array)[1],2)-round(min(x)/np.shape(gt_array)[1],2):
    #     xlenmin = round(max(x)/np.shape(gt_array)[1],2)-round(min(x)/np.shape(gt_array)[1],2)
    # if ylenmin > round(max(y)/np.shape(gt_array)[1],2)-round(min(y)/np.shape(gt_array)[1],2):
    #     ylenmin = round(max(y)/np.shape(gt_array)[1],2)-round(min(y)/np.shape(gt_array)[1],2)
    if gt_array.shape[1] == 480 or  gt_array.shape[1] == 512:
        data_array = data_array[:,136:360,136:360]
        gt_array = gt_array[:,136:360,136:360]
    else:
        print("error:",gt_array.shape)
    
    # show_img(gt_array)
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

#    for iii in range(np.shape(data_array)[0]):
#        scipy.misc.imsave(img_dir+'mask_pat_'+str(pp)+'_'+str(iii)+'.png', mask[iii, ...])
#        scipy.misc.imsave(img_dir+'img_pat_'+str(pp)+'_'+str(iii)+'.png', data_array_[iii, ...])
#        scipy.misc.imsave(img_dir+'gt_pat_'+str(pp)+'_'+str(iii)+'.png', gt_array_[iii, ...])
#LGE_data_1ch = np.array(LGE_data_1ch)
#LGE_gt_1ch = np.array(LGE_gt_1ch)
LGE_data_1ch = np.asarray(LGE_data_1ch)
LGE_gt_1ch = np.asarray(LGE_gt_1ch)
LGE_gt_1ch[LGE_gt_1ch == 500] = 1
LGE_gt_1ch[LGE_gt_1ch == 200] = 2
LGE_gt_1ch[LGE_gt_1ch == 600] = 3
np.save('LGE_data_1ch.npy', LGE_data_1ch)
np.save('LGE_gt_1ch.npy', LGE_gt_1ch)
# print(xmin,xmax,ymin,ymax, xlenmin, ylenmin)
# xmin = 1 
# xmax = 1
# ymin = 1
# ymax = 1
# xlenmin = 1
# ylenmin = 1
##### T2
T2_data_1ch = []
T2_gt_1ch = []
img_dir = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/t2_images/'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
gt_dir_1 = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/t2gt/'

for pp in range(1, 31):
    data_name = data_dir + 'patient' + str(pp) + '_T2.nii.gz'
    gt_name = gt_dir_1 + 'patient' + str(pp) + '_T2_manual.nii.gz'

    data_array = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(data_name)))
    gt_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_name)))
    data_array = np.nan_to_num(data_array, copy=True)
    gt_array = np.nan_to_num(gt_array, copy=True)
    print(gt_array.shape)
    img_count +=gt_array.shape[0]
    # count = 0
    # for image in data_array:
    #     new_image = resize(image, (480,480), anti_aliasing=True)
    #     if count == 0:
    #             new_data_array = new_image[np.newaxis,:,:]
    #     else:
    #             new_data_array = np.concatenate((new_data_array, new_image[np.newaxis,:,:]), axis=0)

    #     count += 1
    # data_array = new_data_array
    # new_gt_array = 0
    # count = 0
    # for gt in gt_array:
    #     new_gt = resize(gt, (480,480), anti_aliasing=True)
    #     if count == 0:
    #             new_gt_array = new_gt[np.newaxis,:,:]
    #     else:
    #             new_gt_array = np.concatenate((new_gt_array, new_gt[np.newaxis,:,:]), axis=0)

    #     count += 1
    # gt_array = new_gt_array.astype(int)
    x = []
    y = []
    count = 0
    print("idx:", pp)
    for image in gt_array:
        for i in range(np.shape(gt_array)[1]):
            for j in range(np.shape(gt_array)[2]):
                if image[i][j] != 0:
                    if j < 30 or i < 30:
                        # show_img(image.shape)
                        gt_array[count, 0:75, 0:50] = 0
                    else:
                        x.append(i)
                        y.append(j)
                    
        count += 1

    
    print(min(x), max(x),
          max(x) - min(x), round(min(x) / np.shape(gt_array)[1], 2),
          round(max(x) / np.shape(gt_array)[1], 2))
    print(min(y), max(y),
          max(y) - min(y), round(min(y) / np.shape(gt_array)[1], 2),
          round(max(y) / np.shape(gt_array)[1], 2))
    if(round(min(x)/np.shape(gt_array)[1],2) < 0.2 or round(min(y)/np.shape(gt_array)[1],2)<0.2):
        print("errorerrorerrorerrorerrorerror")
        show_img(gt_array)
    if int(gt_array.shape[1]) == 256:
        data_array = data_array[:,16:240,16:240]
        gt_array = gt_array[:,16:240,16:240]
    elif gt_array.shape[1] == 288:
        data_array = data_array[:,32:256,32:256]
        gt_array = gt_array[:,32:256,32:256]
    elif gt_array.shape[1] == 240:
        data_array = data_array[:,8:232,8:232]
        gt_array = gt_array[:,8:232,8:232]
    elif gt_array.shape[1] == 224:
        pass
    else:
        print("error:",gt_array.shape)
    
    # if xmin > round(min(x)/np.shape(gt_array)[1],2):
    #     xmin = round(min(x)/np.shape(gt_array)[1],2)
    # if xmax > round(max(x)/np.shape(gt_array)[1],2):
    #     xmax = round(max(x)/np.shape(gt_array)[1],2)
    # if ymin > round(min(y)/np.shape(gt_array)[1],2):
    #     ymin = round(min(y)/np.shape(gt_array)[1],2)
    # if ymax > round(max(y)/np.shape(gt_array)[1],2):
    #     ymax = round(max(y)/np.shape(gt_array)[1],2)
    # if xlenmin > round(max(x)/np.shape(gt_array)[1],2)-round(min(x)/np.shape(gt_array)[1],2):
    #     xlenmin = round(max(x)/np.shape(gt_array)[1],2)-round(min(x)/np.shape(gt_array)[1],2)
    # if ylenmin > round(max(y)/np.shape(gt_array)[1],2)-round(min(y)/np.shape(gt_array)[1],2):
    #     ylenmin = round(max(y)/np.shape(gt_array)[1],2)-round(min(y)/np.shape(gt_array)[1],2) 
    mask = np.zeros(np.shape(data_array), dtype='float32')
    mask[data_array >= thresh] = 1
    mask[data_array < thresh] = 0
    # print("------------------")
    # print("mask1:",data_array >= thresh)
    # print("mask2:",data_array < thresh)
    # print("------------------")
    # time.sleep()
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

    print("np.max(data_array_):",np.max(data_array_))
    T2_data_1ch.extend(np.float32(data_array_))
    T2_gt_1ch.extend(np.float32(gt_array_))

    for iii in range(np.shape(data_array)[0]):
        scipy.misc.imsave(
            img_dir + 'mask_pat_' + str(pp) + '_' + str(iii) + '.png',
            mask[iii, ...])
        scipy.misc.imsave(
            img_dir + 'img_pat_' + str(pp) + '_' + str(iii) + '.png',
            data_array_[iii, ...])
        scipy.misc.imsave(
            img_dir + 'gt_pat_' + str(pp) + '_' + str(iii) + '.png',
            gt_array_[iii, ...])

#T2_data_1ch_ = np.zeros([np.shape(T2_data_1ch)[0], rows, cols])
#T2_gt_1ch_ = np.zeros([np.shape(T2_data_1ch)[0], rows, cols])
#for iii in range(0, np.shape(T2_data_1ch)[0]):
#    T2_data_1ch_[iii, ...] = T2_data_1ch[iii]
#    T2_gt_1ch_[iii, ...] = T2_gt_1ch[iii]
T2_data_1ch = np.asarray(T2_data_1ch)
T2_gt_1ch = np.asarray(T2_gt_1ch)
T2_gt_1ch[T2_gt_1ch == 500] = 1
T2_gt_1ch[T2_gt_1ch == 200] = 2
T2_gt_1ch[T2_gt_1ch == 600] = 3
np.save('T2_data_1ch.npy', T2_data_1ch)
np.save('T2_gt_1ch.npy', T2_gt_1ch)
# print(xmin,xmax,ymin,ymax, xlenmin, ylenmin)
# xmin = 1 
# xmax = 1
# ymin = 1
# ymax = 1
# xlenmin = 1
# ylenmin = 1
#######C0
#
C0_data_1ch = []
C0_gt_1ch = []
img_dir = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0_images/'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
gt_dir_1 = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0gt/'

for pp in range(1, 31):
    data_name = data_dir + 'patient' + str(pp) + '_C0.nii.gz'
    gt_name = gt_dir_1 + 'patient' + str(pp) + '_C0_manual.nii.gz'
    data_array = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(data_name)))
    gt_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_name)))
    print(np.shape(data_array))
    img_count +=gt_array.shape[0]
#     new_data_array = 0
#     count = 0
#     for image in data_array:
#         new_image = resize(image, (480,480), anti_aliasing=True)
#         if count == 0:
#                 new_data_array = new_image[np.newaxis,:,:]
#         else:
#                 new_data_array = np.concatenate((new_data_array, new_image[np.newaxis,:,:]), axis=0)

#         count += 1
#     data_array = new_data_array
# #     show_img(new_data_array)
#     new_gt_array = 0
#     count = 0
#     for gt in gt_array:
#         new_gt = resize(gt, (480,480), anti_aliasing=True)
#         if count == 0:
#                 new_gt_array = new_gt[np.newaxis,:,:]
#         else:
#                 new_gt_array = np.concatenate((new_gt_array, new_gt[np.newaxis,:,:]), axis=0)

#         count += 1
#     gt_array = new_gt_array.astype(int)
    x = []
    y = []
    for image in gt_array:
        for i in range(np.shape(gt_array)[1]):
            for j in range(np.shape(gt_array)[2]):
                if image[i][j] != 0:
                    if i < 30 or j <30:
                        print("label_error:", pp,image.shape)
                    else:
                        x.append(i)
                        y.append(j)
    print("idx:", pp)
    print(min(x), max(x),
          max(x) - min(x), round(min(x) / np.shape(gt_array)[1], 2),
          round(max(x) / np.shape(gt_array)[1], 2))
    print(min(y), max(y),
          max(y) - min(y), round(min(y) / np.shape(gt_array)[1], 2),
          round(max(y) / np.shape(gt_array)[1], 2))
    if gt_array.shape[1] == 320:
        data_array = data_array[:,64:288,64:288]
        gt_array = gt_array[:,64:288,64:288]
    elif gt_array.shape[1] == 288:
        data_array = data_array[:,32:256,32:256]
        gt_array = gt_array[:,32:256,32:256]
    elif gt_array.shape[1] == 240:
        data_array = data_array[:,8:232,8:232]
        gt_array = gt_array[:,8:232,8:232]
    elif gt_array.shape[1] == 256:
        data_array = data_array[:,16:240,16:240]
        gt_array = gt_array[:,16:240,16:240]
    elif gt_array.shape[1] == 224:
        pass
    else:
        print("error:",gt_array.shape)
    # if(round(min(x)/np.shape(gt_array)[1],2) < 0.2 or round(min(y)/np.shape(gt_array)[1],2)<0.2):
    #     show_img(gt_array)      
    # if xmin > round(min(x)/np.shape(gt_array)[1],2):
    #     xmin = round(min(x)/np.shape(gt_array)[1],2)
    # if xmax > round(max(x)/np.shape(gt_array)[1],2):
    #     xmax = round(max(x)/np.shape(gt_array)[1],2)
    # if ymin > round(min(y)/np.shape(gt_array)[1],2):
    #     ymin = round(min(y)/np.shape(gt_array)[1],2)
    # if ymax > round(max(y)/np.shape(gt_array)[1],2):
    #     ymax = round(max(y)/np.shape(gt_array)[1],2)
    # if xlenmin > round(max(x)/np.shape(gt_array)[1],2)-round(min(x)/np.shape(gt_array)[1],2):
    #     xlenmin = round(max(x)/np.shape(gt_array)[1],2)-round(min(x)/np.shape(gt_array)[1],2)
    # if ylenmin > round(max(y)/np.shape(gt_array)[1],2)-round(min(y)/np.shape(gt_array)[1],2):
    #     ylenmin = round(max(y)/np.shape(gt_array)[1],2)-round(min(y)/np.shape(gt_array)[1],2) 
    mask = np.zeros(np.shape(data_array), dtype='float32')
    mask[data_array >= thresh] = 1
    mask[data_array < thresh] = 0
    # print(data_array >= thresh)

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
    C0_data_1ch.extend(np.float32(data_array_))
    C0_gt_1ch.extend(np.float32(gt_array_))

    for iii in range(np.shape(data_array)[0]):
        scipy.misc.imsave(
            img_dir + 'mask_pat_' + str(pp) + '_' + str(iii) + '.png',
            mask[iii, ...])
        scipy.misc.imsave(
            img_dir + 'img_pat_' + str(pp) + '_' + str(iii) + '.png',
            data_array_[iii, ...])
        scipy.misc.imsave(
            img_dir + 'gt_pat_' + str(pp) + '_' + str(iii) + '.png',
            gt_array_[iii, ...])

C0_data_1ch = np.asarray(C0_data_1ch)
C0_gt_1ch = np.asarray(C0_gt_1ch)
C0_gt_1ch[C0_gt_1ch == 500] = 1
C0_gt_1ch[C0_gt_1ch == 200] = 2
C0_gt_1ch[C0_gt_1ch == 600] = 3
# np.save('C0_data_1ch.npy', C0_data_1ch)
# np.save('C0_gt_1ch.npy', C0_gt_1ch)

new_data_array = np.concatenate((LGE_data_1ch, C0_data_1ch), axis=0)
new_data_array = np.concatenate((new_data_array, T2_data_1ch), axis=0)
new_gt_array = np.concatenate((LGE_gt_1ch, C0_gt_1ch), axis=0)
new_gt_array = np.concatenate((new_gt_array, T2_gt_1ch), axis=0)
np.save('train_data.npy', new_data_array[:, :, :, np.newaxis])
np.save('train_gt.npy', new_gt_array[:, :, :, np.newaxis])
print("img_count:",img_count)
print("new_gt_array:",new_gt_array.shape)
# print(xmin,xmax,ymin,ymax, xlenmin, ylenmin)
