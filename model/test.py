from keras.optimizers import RMSprop, Adam
from models.CNN import build_discriminator
from models.DRUNet32f import get_model
from keras.models import Model
from keras.layers import Input,Concatenate

import numpy as np
import SimpleITK as sitk

img_shape = (224, 224, 1)
masks_shape= (224, 224, 4)
num_classes = 4
learn_rate = 2e-4
learn_decay = 1e-8
orig_num = 15
orig_rows = 512
orig_cols = 512
output_file = "test_6_cnn_pred_epoch_60.nii.gz"
weight_file = "adversarial_weights_epoch_60.h5"
test_file = "LGE_data_1ch_extra.npy"
orig_mask_1 = np.zeros([orig_num, orig_rows, orig_cols], dtype = 'float32')

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


test_images = np.load(test_file)
test_images = test_images[:15,:,:, np.newaxis]

pred_masks_1 = adversarial_model.predict(test_images)
pred_masks_1 = pred_masks_1[0].argmax(axis=3)
rows = np.shape(pred_masks_1)[1]
cols = np.shape(pred_masks_1)[2]
orig_mask_1[:, int((orig_rows-rows)/2):int((orig_rows-rows)/2)+rows, int((orig_cols-cols)/2):int((orig_cols-cols)/2)+cols] = pred_masks_1
sitk.WriteImage(sitk.GetImageFromArray(orig_mask_1),output_file)

