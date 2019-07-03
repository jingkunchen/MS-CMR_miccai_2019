from keras.optimizers import RMSprop, Adam
from models.CNN import build_discriminator
from models.DRUNet32f import get_model
from keras.models import Model
from keras.layers import Input,Concatenate
from skimage.transform import resize

import numpy as np
import SimpleITK as sitk
import skimage.io as io
import os 
import re
img_shape = (256, 256, 1)
masks_shape= (256, 256, 4)
resize_shape = (256, 256)
num_classes = 4
learn_rate = 2e-4
learn_decay = 1e-8
test_file = "/Users/chenjingkun/Documents/code/python/MS-CMR_miccai_2019/model/resize_lge_data_256_256.npy"
patient_path = "/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0t2lge"

# argumentation
weight_file = "/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/del/adversarial_weights_epoch_20.h5"
output_path = "/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/del/test_patient_cnn_pred_epoch_20.nii.gz"

opt_discriminator = Adam(lr=(learn_rate))

mask_shape_discrimator = (masks_shape[0], masks_shape[1],
                        num_classes + 1)
                        
optimizer = RMSprop(lr=learn_rate, clipvalue=1.0, decay= learn_decay)

discriminator = build_discriminator(mask_shape_discrimator)
discriminator.compile(loss='binary_crossentropy', optimizer=opt_discriminator)
discriminator.summary()
generator_nn = get_model(img_shape=img_shape, num_classes=num_classes)
generator_nn.compile(loss='categorical_crossentropy',
                    optimizer=opt_discriminator)
img = Input(shape=img_shape)
rec_mask = generator_nn(img)
rec_mask_new = Concatenate()([rec_mask, img])
discriminator.trainable = False
validity = discriminator(rec_mask_new)
adversarial_model = Model(img, [rec_mask, validity], name='D')
adversarial_model.compile(
    loss=['categorical_crossentropy', 'binary_crossentropy'],
    loss_weights=[1, 1],
    optimizer=optimizer)

adversarial_model.load_weights(weight_file)

 
def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data.shape

def show_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    for i in range(data.shape[0]):
        io.imshow(data[i,:,:], cmap = 'gray')
        print(i)
        io.show()

def get_shape_dict():
    shape_dict = {}
    for root, dirs, files in os.walk(patient_path):
            for i in files:
                if (None != re.search("_LGE.nii.gz", i)):
                    tmp = i.replace("_LGE.nii.gz","")
                    idx = tmp.replace("patient","")
                    print("idx:", idx)
                    shape = read_img(os.path.join(patient_path, i))
                    shape_dict[int(idx)]= shape
    return shape_dict

def predict(orig_num, orig_rows, orig_cols, output_file, start, end):
    print("orig_num, orig_rows, orig_cols, output_file, start, end:",orig_num, orig_rows, orig_cols, output_file, start, end)
    orig_mask_1 = np.zeros([orig_num, orig_rows, orig_cols], dtype = 'float32')

    test_images = np.load(test_file)
    
    print("test_images:",test_images.shape)
    test_images = test_images[start:end,:,:,np.newaxis]

    pred_masks_1 = adversarial_model.predict(test_images)
    pred_masks_1 = pred_masks_1[0].argmax(axis=3)
    rows = np.shape(pred_masks_1)[1]
    cols = np.shape(pred_masks_1)[2]
    orig_mask_1[:, int((orig_rows-rows)/2):int((orig_rows-rows)/2)+rows, int((orig_rows-cols)/2):int((orig_rows-cols)/2)+cols] = pred_masks_1
    # new_gt_list = []
    # for image in orig_mask_1:        
    #     image = np.asarray(image)
    #     image1 = image.copy()
    #     image2 = image.copy()
    #     image[image == 500] = 1
    #     image[image == 200] = 0
    #     image[image == 600] = 0
    #     image1[image1 == 500] = 0
    #     image1[image1 == 200] = 1
    #     image1[image1 == 600] = 0
    #     image2[image2 == 500] = 0
    #     image2[image2 == 200] = 0
    #     image2[image2 == 600] = 1

    #     image = resize(image,(orig_rows,orig_cols), preserve_range =True)
    #     image1 = resize(image1,(orig_rows,orig_cols), preserve_range =True)
    #     image2 = resize(image2,(orig_rows,orig_cols), preserve_range =True)
    #     image = np.around(image)
    #     image1 = np.around(image1)
    #     image2 = np.around(image2)
    #     image = image.astype(np.int32)
    #     image1 = image1.astype(np.int32)
    #     image2 = image2.astype(np.int32)

    #     image[image == 1] = 1
    #     image1[image1 == 1] = 2
    #     image2[image2 == 1] = 3
    #     image = image +image1 +image2
    #     [x_test, y_test] = image.shape
    #     for i in range(x_test):
    #         for j in range(y_test):
    #             if(image[i, j] >3) :
    #                 # print("--------error----------:",image[i, j])
    #                 # image[i, j] = int(image[i, j]/3)
    #                 # image[i, j] = 0
    #                 pass

    #     image[image == 1] = 500
    #     image[image == 2] = 200
    #     image[image == 3] = 600
    #     for i in range(image.shape[0]):
    #         for j in range(image.shape[1]):
    #             if image[i][j] != 0:
    #                 if j < 40 or i < 40:
    #                     image[0:200, 0:50] = 0
               
    #     new_gt_list.append(image)
                    
    # orig_mask_1=np.array(new_gt_list)
    # return orig_mask_1
    sitk.WriteImage(sitk.GetImageFromArray(orig_mask_1),output_file)

def main():
    test_images = np.load(test_file)
    print("test_images:",test_images.shape)
    shape_dict = get_shape_dict()
    slice_count = -1
    tmp1 = None
    for i in range(6,46):
        print(i)
        tmp = shape_dict[i]
        orig_num = tmp[0]
        orig_rows = tmp[1]
        orig_cols = tmp[2]
       
        print("shape:", orig_num, orig_rows, orig_cols)
        output_file = output_path.replace("patient",str(i))
        
        start = slice_count + 1
        print("start:",start)
        slice_count = slice_count + orig_num
        end = slice_count + 1
        print("end:",end)
        predict(orig_num, orig_rows, orig_cols, output_file, start, end)
        # result = predict(orig_num, orig_rows, orig_cols, output_file, start, end)
        # print("result:",result.shape)
        # if tmp1 is None:
        #     tmp1 = result
        #     print(tmp1.shape)
        # else:
        #     print("tmp1:",tmp1.shape)
        #     print("result:",result.shape)
        #     tmp1 = np.concatenate((tmp1, result),axis = 0)
        #     print(tmp1.shape)

    
        
    print("slice_count:",slice_count)

if __name__ == "__main__":
    main()