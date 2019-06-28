from __future__ import print_function
import os
import numpy as np
import SimpleITK as sitk
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
 
data_dir = '/home/bran/CODES/Cardiac_segmentation_challenge/C0LET2_nii45_for_challenge19/c0t2lge/'
thresh = 1
rows = 224
cols = 224
 
# label transform, 500-->1, 200-->2, 600-->3
 
###### LGE
LGE_data_1ch = []
LGE_gt_1ch = [] 
img_dir = '/home/bran/CODES/Cardiac_segmentation_challenge/C0LET2_nii45_for_challenge19/LGE_images/'
if not os.path.exists(img_dir):
        os.makedirs(img_dir)
gt_dir_1 = '/home/bran/CODES/Cardiac_segmentation_challenge/C0LET2_nii45_for_challenge19/lgegt/'
 
for pp in range(6, 46):
 
    data_name = data_dir+'patient'+str(pp)+'_LGE.nii.gz'
    data_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_name)))
    print(np.shape(data_array))
     
    mask = np.zeros(np.shape(data_array), dtype = 'float32')
    mask[data_array >=thresh] = 1
    mask[data_array < thresh] = 0
    for iii in range(np.shape(data_array)[0]):
        mask[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(mask[iii,:,:])  #fill the holes inside br
    data_array = data_array - np.mean(data_array[mask == 1])
    data_array /= np.std(data_array[mask == 1])    
    rows_o = np.shape(data_array)[1]
    cols_o = np.shape(data_array)[2]
     
    data_array_ = data_array[:, int((rows_o-rows)/2):int((rows_o-rows)/2)+rows, int((cols_o-cols)/2):int((cols_o-cols)/2)+cols]
    LGE_data_1ch.extend(np.float32(data_array_))
 
LGE_data_1ch = np.asarray(LGE_data_1ch)
np.save('LGE_data_1ch_extra.npy',LGE_data_1ch)
 