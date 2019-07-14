from keras.optimizers import RMSprop, Adam
from models.CNN import build_discriminator
from models.DRUNet32f import get_model
from keras.models import Model
from keras.layers import Input,Concatenate

import numpy as np
import SimpleITK as sitk
import skimage.io as io
import os
import re
import time
from keras import backend as K
from keras.losses import categorical_crossentropy
from skimage.morphology import square
from skimage.morphology import opening,closing,erosion,dilation
from keras.utils.vis_utils import plot_model

img_shape = (224, 224, 1)
masks_shape= (224, 224, 4)
num_classes = 4
learn_rate = 2e-4
learn_decay = 1e-8
test_file = "/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/dice/LGE_test_224_224.npy"
patient_path = "/Users/chenjingkun/Documents/data/C0LET2_nii45_for_challenge19/c0t2lge"

weight_file = "/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/test/arg_adversarial_weights_epoch_30.h5"
output_path = "/Users/chenjingkun/Documents/result/MS-CMR_miccai_2019_result/test/patient_30.nii.gz"

smooth=1.
def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def cross_dice_coef_loss(y_true, y_pred):
    x0 = 1.-dice_coef_for_training(1-y_true[0], 1-y_pred[0])
    x1 = 1.-dice_coef_for_training(y_true[1], y_pred[1])
    x2 = 1.-dice_coef_for_training(y_true[2], y_pred[2])
    x3 = 1.-dice_coef_for_training(y_true[3], y_pred[3])
    return x0+x1+x2+x3

def dice_coef_loss(y_true, y_pred):
    x = 1.-dice_coef_for_training(y_true, y_pred)
    return x

def dice_cross_loss(y_true, y_pred):
    return 0.8*categorical_crossentropy(y_true,y_pred) + 0.2*dice_coef_loss(y_true, y_pred)

loss_function = dice_cross_loss
opt_discriminator = Adam(lr=(learn_rate))

mask_shape_discrimator = (masks_shape[0], masks_shape[1],
                        num_classes + 1)

optimizer = RMSprop(lr=learn_rate, clipvalue=1.0, decay= learn_decay)

discriminator = build_discriminator(mask_shape_discrimator)
print("discrimator")
discriminator.compile(loss='binary_crossentropy', optimizer=opt_discriminator)
discriminator.summary()
plot_model(discriminator, to_file='discriminator.png',show_shapes=True)
generator_nn = get_model(img_shape=img_shape, num_classes=num_classes)
generator_nn.compile(loss=dice_cross_loss,
                    optimizer=opt_discriminator)
plot_model(generator_nn, to_file='generator.png',show_shapes=True)

print("generator")                    
generator_nn.summary()
img = Input(shape=img_shape)
rec_mask = generator_nn(img)
rec_mask_new = Concatenate()([rec_mask, img])
discriminator.trainable = False
validity = discriminator(rec_mask_new)
adversarial_model = Model(img, [rec_mask, validity], name='D')
adversarial_model.compile(
    loss=[dice_cross_loss, 'binary_crossentropy'],
    loss_weights=[1, 1],
    optimizer=optimizer)
plot_model(adversarial_model, to_file='adversarial.png',show_shapes=True)

print("adversarial_model")                    
adversarial_model.summary()
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
    # orig_mask_1 = np.zeros([orig_num, orig_rows, orig_cols], dtype = 'float32')
    orig_mask_1 = np.zeros([orig_num, 224, 224], dtype = 'float32')

    test_images = np.load(test_file)
    print("test_images:",test_images.shape)
    test_images = test_images[start:end,:,:,np.newaxis]
    pred_masks_1 = adversarial_model.predict(test_images)
    pred_masks_1 = pred_masks_1[0].argmax(axis=3)
    rows = np.shape(pred_masks_1)[1]
    cols = np.shape(pred_masks_1)[2]
    # print("rows:",rows)
    # print("cols:",cols)
    # print("rows:",int((orig_rows-rows)/2),int((orig_rows-rows)/2)+rows)
    # print("cols:",int((orig_cols-cols)/2),int((orig_cols-cols)/2)+cols)
    # orig_mask_1[:, int((orig_rows-rows)/2):int((orig_rows-rows)/2)+rows, int((orig_cols-cols)/2):int((orig_cols-cols)/2)+cols] = pred_masks_1
    orig_mask_1[:,:,:] = pred_masks_1

    # new_gt_list = [] 
    # for image in orig_mask_1:
    #     image = closing(image, square(3))
    #     image = opening(image, square(3))
    #     # image = erosion(image, square(3))
    #     new_gt_list.append(image)            
    # orig_mask_1=np.array(new_gt_list)
    # orig_mask_1 = np.asarray(orig_mask_1)
    # orig_mask_1[orig_mask_1 == 1] = 500
    # orig_mask_1[orig_mask_1 == 2] = 200
    # orig_mask_1[orig_mask_1 == 3] = 600
    
      # if orig_rows == 480 or orig_rows == 512:
    #     orig_mask_1[:, 136:360, 136:360] = pred_masks_1
    # elif int(orig_rows) == 400:
    #     orig_mask_1[:, 88:312, 88:312] = pred_masks_1
    #     # data_array = data_array[:,88:312,88:312]
    # elif int(orig_rows) == 432:
    #     orig_mask_1[:, 104:328, 104:328] = pred_masks_1
    #     # data_array = data_array[:,104:328,104:328]
    # elif orig_rows == 224:
    #     orig_mask_1 = pred_masks_1
    # else:
    #     print("error:",orig_rows, orig_rows == 400)
    return orig_mask_1
    # sitk.WriteImage(sitk.GetImageFromArray(orig_mask_1),output_file)

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
        result =  predict(orig_num, orig_rows, orig_cols, output_file, start, end)
        if tmp1 is None:
            tmp1 = result.copy()
            print("start:",i,tmp1.shape)
        else:
            print("start:",i,tmp1.shape)
            tmp1 = np.concatenate((tmp1, result),axis = 0)
        print("end:",i,tmp1.shape)
    # print("total:",tmp1.shape)
   
    sitk.WriteImage(sitk.GetImageFromArray(tmp1),output_path)

    


    print("slice_count:",slice_count)

if __name__ == "__main__":
    main()