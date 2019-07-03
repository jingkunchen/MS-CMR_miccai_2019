from __future__ import print_function
import os
import numpy as np
import SimpleITK as sitk
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage
import cv2
import time
from decimal import Decimal
import skimage.io as io
from skimage.morphology import square
from skimage.morphology import dilation
from skimage.transform import resize


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
    for i in range(data.shape[0]):
        io.imshow(data[i, :, :], cmap='gray')
        # io.imshow(data[:,:], cmap = 'gray')
        io.show()
def show_img_single(data):
    # for i in range(data.shape[0]):
        # io.imshow(data[i, :, :], cmap='gray')
        io.imshow(data[:,:], cmap = 'gray')
        io.show()


# label transform, 500-->1, 200-->2, 600-->3

###### LGE
LGE_data_1ch = []
# LGE_gt_1ch = []
img_dir = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/lge_images/'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
gt_dir_1 = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/reclgegt/'
for pp in range(6, 46):

    data_name = data_dir + 'patient' + str(pp) + '_LGE.nii.gz'
    # gt_name = gt_dir_1 + 'new_data_test_' + str(pp) + '_cnn_pred_epoch_110.nii.gz'
    # img = sitk.ReadImage(os.path.join(gt_name))
    data_array = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(data_name)))
    # gt_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_name)))
    
    data_array = data_array.astype(int)
    img_count +=data_array.shape[0]
    # show_img(gt_array)
    print(np.shape(data_array))
    # print(np.shape(gt_array))

    x = []
    y = []
    print("idx:", pp)
    # for image in gt_array:
    #     # tmp = dilation(image, square(3))
    #     # show_img_single(tmp)
    #     for i in range(np.shape(gt_array)[1]):
    #         for j in range(np.shape(gt_array)[2]):
    #             if image[i][j] != 0:
    #                 if i <30 or j<30:
    #                     print("label_error:", pp,i,j,image[i][j])
    #                 else:
    #                     x.append(i)
    #                     y.append(j)
    # print(min(x),max(x),max(x)-min(x),round(min(x)/np.shape(gt_array)[1],2), round(max(x)/np.shape(gt_array)[1],2))
    # print(min(y),max(y),max(y)-min(y),round(min(y)/np.shape(gt_array)[1],2), round(max(y)/np.shape(gt_array)[1],2))
    # 1
    # if data_array.shape[1] == 480 or data_array.shape[1] == 512:
    #     data_array = data_array[:,136:360,136:360]
    # elif int(data_array.shape[1]) == 400:
    #     data_array = data_array[:,88:312,88:312]
    # elif int(data_array.shape[1]) == 432:
    #     data_array = data_array[:,104:328,104:328]
    # elif data_array.shape[1] == 224:
    #     pass
    # else:
    #     print("error:",data_array.shape, int(data_array.shape[1]) == 400)
    # 1 
    # if data_array.shape[1] == 480 or  data_array.shape[1] == 512:
    #     # 1
    #     # data_array = data_array[:,136:360,136:360]
    #     # gt_array = gt_array[:,136:360,136:360]
    #     # 2
    #     data_array = data_array[:,120:344,120:344]
    # elif int(data_array.shape[1]) == 400:
    #     data_array = data_array[:,70:294,70:294]
    # elif int(data_array.shape[1]) == 432:
    #     data_array = data_array[:,96:330,96:330]
    # elif data_array.shape[1] == 224:
    #     pass
    # else:
    #     print("error:",data_array.shape, int(data_array.shape[1]) == 400)
    
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
    # gt_array_ = gt_array[:,
    #                      int((rows_o - rows) /
    #                          2):int((rows_o - rows) / 2) + rows,
    #                      int((cols_o - cols) / 2):int((cols_o - cols) / 2) +
    #                      cols]
  

    LGE_data_1ch.extend(np.float32(data_array_))
    # LGE_gt_1ch.extend(np.float32(gt_array_))


LGE_data_1ch = np.asarray(LGE_data_1ch)
# LGE_gt_1ch = np.asarray(LGE_gt_1ch)
# LGE_gt_1ch[LGE_gt_1ch == 500] = 1
# LGE_gt_1ch[LGE_gt_1ch == 200] = 2
# LGE_gt_1ch[LGE_gt_1ch == 600] = 3
output_path = "/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/del/LGE_data_1ch_extra_256_256.nii.gz"
sitk.WriteImage(sitk.GetImageFromArray(LGE_data_1ch),output_path)
np.save('LGE_data_1ch_extra_256_256.npy', LGE_data_1ch)
# np.save('LGE_gt_1ch_extra.npy', LGE_gt_1ch)
# print("LGE_gt_1ch:",LGE_gt_1ch.shape)
print(img_count)