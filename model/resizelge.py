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
from skimage.morphology import square
from skimage.morphology import dilation

new_shape = (480,480)
new_shape_1 = (480, 480)
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
        io.show()
def show_img_single(data):
        io.imshow(data[:,:], cmap = 'gray')
        io.show()

# label transform, 500-->1, 200-->2, 600-->3

###### LGE
LGE_data_1ch = []
img_dir = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/lge_images/'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
for pp in range(6, 46):

    data_name = data_dir + 'patient' + str(pp) + '_LGE.nii.gz'
    data_array = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(data_name)))
    
    img_count +=data_array.shape[0]
    print(np.shape(data_array))

    x = []
    y = []
    print("idx:", pp)
    new_data_list = []
    for image in data_array:
        image = np.asarray(image)
        image = resize(image, new_shape_1, preserve_range =True)
        image = np.around(image)
        image = image.astype(np.int32)
        new_data_list.append(image)
    data_array=np.array(new_data_list)
    print("tmp:",data_array.shape)   

    
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
    new_data_list = []


    data_array_ = data_array[:,
                             int((rows_o - rows) /
                                 2):int((rows_o - rows) / 2) + rows,
                             int((cols_o - cols) /
                                 2):int((cols_o - cols) / 2) + cols]
   

    LGE_data_1ch.extend(np.float32(data_array_))


LGE_data_1ch = np.asarray(LGE_data_1ch)
print("LGE_data_1ch:",LGE_data_1ch.shape)

output_path = "/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/del/resize_lge_data_256_256.nii.gz"

sitk.WriteImage(sitk.GetImageFromArray(LGE_data_1ch),output_path)
np.save('resize_lge_data_256_256.npy', LGE_data_1ch)
# print(img_count)