import os
import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
import skimage.io as io
import scipy.misc

def show_img(data):
    # for i in range(data.shape[0]):
    #     io.imshow(data[i, :, :], cmap='gray')
        io.imshow(data[:,:], cmap = 'gray')
        io.show()

rows = 224
cols = 224
start_number = 6
end_number = 36

thresh = 1
data_dir = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0t2lge/'
T2_gt_1ch = []
LGE_shape = []
T2_shape = []

gt_dir_1 = '/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0gt/'
lge_count = 0

for pp in range(start_number, end_number):

    data_name = data_dir + 'patient' + str(pp) + '_LGE.nii.gz'
    data_array = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(data_name)))
    data_array = np.nan_to_num(data_array, copy=True)
    lge_count +=data_array.shape[0]
    LGE_shape.append(data_array.shape)
    print("LGE_shape:",data_array.shape)
    # print("LGE_shape:",LGE_shape[pp-6][0])
    
    print("lge_count:",lge_count)

T2_count = 0
for pp in range(start_number, end_number):
    new_shape = (LGE_shape[pp-start_number][1],LGE_shape[pp-start_number][2])
    gt_name = gt_dir_1 + 'patient' + str(pp) + '_C0_manual.nii.gz'
    gt_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_name)))

    gt_array = np.nan_to_num(gt_array, copy=True)
    print(gt_array.shape)
    T2_shape.append(gt_array.shape)
    T2_count +=gt_array.shape[0]


LGE_data_1ch = []
img_count = 0
new_count_x_list = []
new_count_y_list = []
for pp in range(start_number, end_number):
    lge_number = LGE_shape[pp-start_number][0]
    t2_number = T2_shape[pp-start_number][0]
    print("lge_number:",lge_number)
    print("t2_number:",t2_number)
    

    data_name = data_dir + 'patient' + str(pp) + '_LGE.nii.gz'
    data_array = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(data_name)))
    
    img_count +=data_array.shape[0]
    print(np.shape(data_array))
    
        
    slice_number = []
    for i in range(t2_number):
        slice_number.append(round(float(lge_number)/float(t2_number)*(i+1)))
    print("slice_number:",slice_number)
    count_i = 0
    count = 0
    print("data_array:",data_array[slice_number[0]-1,:,:].shape)
    new_data_array_list = []
    for i in slice_number:
       new_data_array_list.append(data_array[i-1,:,:])
    data_array=np.array(new_data_array_list)   
    print("tmp:",data_array.shape)        
    

    # for i in range(lge_number):
    #     if i+1 not in slice_number:
    #         delete_number = count_i - count
    #         data_array = np.delete(data_array, delete_number, axis=0)
    #         count += 1
    #     count_i +=1
    
    # if 
    x = []
    y = []
    print("idx:", pp)

    mask = np.zeros(np.shape(data_array), dtype='float32')
    mask[data_array >= thresh] = 1
    mask[data_array < thresh] = 0

    data_array = data_array - np.mean(data_array[mask == 1])
    data_array /= np.std(data_array[mask == 1])
    rows_o = np.shape(data_array)[1]
    cols_o = np.shape(data_array)[2]

    data_array_ = data_array[:,
                             int((rows_o - rows) /
                                 2):int((rows_o - rows) / 2) + rows,
                             int((cols_o - cols) /
                                 2):int((cols_o - cols) / 2) + cols]
    new_count_x_list.append(int((rows_o - rows) /2))
    new_count_y_list.append(int((cols_o - cols) /2))
    
    LGE_data_1ch.extend(np.float32(data_array_))

LGE_data_1ch = np.asarray(LGE_data_1ch)
# sitk.WriteImage(sitk.GetImageFromArray(LGE_data_1ch),"test_image.nii.gz")
# np.save('transfer_T2_data.npy', LGE_data_1ch)

T2_count = 0
for pp in range(start_number, end_number):
    new_count_x = new_count_x_list[pp-start_number]
    new_count_y = new_count_y_list[pp-start_number]
    new_shape = (LGE_shape[pp-start_number][1],LGE_shape[pp-start_number][2])
    gt_name = gt_dir_1 + 'patient' + str(pp) + '_C0_manual.nii.gz'
    gt_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_name)))
    gt_array = np.nan_to_num(gt_array, copy=True)
    print(gt_array.shape)
   
    x = []
    y = []
    count = 0
    print("idx:", pp)
    new_gt_list = []
    for image in gt_array:
        
        image = np.asarray(image)
        # show_img(image)
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
        
        image = resize(image,new_shape, preserve_range =True)
        image1 = resize(image1,new_shape, preserve_range =True)
        image2 = resize(image2,new_shape, preserve_range =True)

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
        [x_test, y_test] = image.shape
        for i in range(x_test):
            for j in range(y_test):
                if(image[i, j] >3) :
                    print("--------error----------:", pp, count)
        image[image == 1] = 500
        image[image == 2] = 200
        image[image == 3] = 600
        
        for i in range(np.shape(gt_array)[1]):
            for j in range(np.shape(gt_array)[2]):
                if image[i][j] != 0:
                    if j < 40 or i < 40:
                        gt_array[count, 0:75, 0:50] = 0
                        image[0:200, 0:50] = 0
                    else:
                        x.append(i)
                        y.append(j)
        new_gt_list.append(image)
        print("new_gt_list:",len(new_gt_list))
                    
        count += 1
    gt_array=np.array(new_gt_list)
    print("new_array:",gt_array.shape)
    
    print(min(x), max(x),
          max(x) - min(x), round(min(x) / np.shape(gt_array)[1], 2),
          round(max(x) / np.shape(gt_array)[1], 2))
    print(min(y), max(y),
          max(y) - min(y), round(min(y) / np.shape(gt_array)[1], 2),
          round(max(y) / np.shape(gt_array)[1], 2))
    if(round(min(x)/np.shape(gt_array)[1],2) < 0.2 or round(min(y)/np.shape(gt_array)[1],2)<0.2):
        print("errorerrorerrorerrorerrorerror")
        show_img(gt_array)
    #C0
    gt_array_ = gt_array[:, new_count_x-4: new_count_x-4 + rows, new_count_y-9: new_count_y-9 + cols]
    #T2
    # gt_array_ = gt_array[:, new_count_x-5: new_count_x-5 + rows, new_count_y-5: new_count_y-5 + cols]


    T2_gt_1ch.extend(np.float32(gt_array_))


T2_gt_1ch = np.asarray(T2_gt_1ch)
T2_gt_1ch[T2_gt_1ch == 500] = 1
T2_gt_1ch[T2_gt_1ch == 200] = 2
T2_gt_1ch[T2_gt_1ch == 600] = 3
sitk.WriteImage(sitk.GetImageFromArray(LGE_data_1ch),"/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/dice/transfer_data_C0_224_224.nii.gz")
np.save('/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/dice/transfer_data_C0_224_224.npy', LGE_data_1ch[:, :, :, np.newaxis])

sitk.WriteImage(sitk.GetImageFromArray(T2_gt_1ch),"/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/dice/transfer_gt_C0_224_224.nii.gz")
np.save('/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/dice/transfer_gt_C0_224_224.npy', T2_gt_1ch)
print(img_count)