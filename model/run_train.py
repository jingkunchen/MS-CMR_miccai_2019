from __future__ import print_function
from __future__ import division

import click
import json
import os
import logging
import numpy as np
import time
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras.layers import Input,Concatenate
from keras.utils import to_categorical
from keras.models import Model
from keras.utils import generic_utils as keras_generic_utils
from keras import backend as K
# from keras.losses import categorical_crossentropy

from models.DRUNet32f import get_model
from models.CNN import build_discriminator
from run_test import get_eval_metrics
from tools.augmentation import augmentation
from tools import patch_utils, logger, facades_generator
from models.discrimator import PatchGanDiscriminator
from models.dcgan import DCGAN
from metrics import weighted_categorical_crossentropy
from keras.optimizers import Adam, RMSprop

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    x0 = 1.-dice_coef_for_training(y_true[0], y_pred[0])
    x1 = 1.-dice_coef_for_training(y_true[1], y_pred[1])
    x2 = 1.-dice_coef_for_training(y_true[2], y_pred[2])
    x3 = 1.-dice_coef_for_training(y_true[3], y_pred[3])
    return x0+x1+x2+x3

@click.command()
@click.argument('train_imgs_0_np_file', type=click.STRING)
@click.argument('train_masks_0_np_file', type=click.STRING)
@click.argument('train_imgs_1_np_file', type=click.STRING)
@click.argument('train_masks_1_np_file', type=click.STRING)
@click.argument('train_imgs_2_np_file', type=click.STRING)
@click.argument('train_masks_2_np_file', type=click.STRING)
@click.argument('output', type=click.STRING)
@click.option('--pretrained_model',
              type=click.STRING,
              default='',
              help='path to the pretrained model')
@click.option('--pretrained_adversarial_model',
              type=click.STRING,
              default='',
              help='path to the pretrained adversarial model')
@click.option('--use_augmentation',
              type=click.BOOL,
              default=False,
              help='use data augmentation or not')
@click.option('--use_patch_gan',
              type=click.BOOL,
              default=False,
              help='use patch gan or not')
@click.option('--use_patch_gan_discrimator',
              type=click.BOOL,
              default=False,
              help='use patch gan image discrimator or not')
@click.option('--use_cnn',
              type=click.BOOL,
              default=False,
              help='use cnn or not')
@click.option('--use_cnn_discrimator',
              type=click.BOOL,
              default=False,
              help='use cnn discrimator or not')
@click.option('--test', type=click.BOOL, default=False, help='test or not')
@click.option('--use_weighted_crossentropy',
              type=click.BOOL,
              default=False,
              help='use weighting of classes according to inbalance or not')
@click.option('--test_imgs_np_file_1',
              type=click.STRING,
              default='',
              help='path to the numpy file of test image')
@click.option('--test_masks_np_file_1',
              type=click.STRING,
              default='',
              help='path to the numpy file of the test image')
@click.option('--test_imgs_np_file_2',
              type=click.STRING,
              default='',
              help='path to the numpy file of test image')
@click.option('--test_masks_np_file_2',
              type=click.STRING,
              default='',
              help='path to the numpy file of the test image')
@click.option('--test_imgs_np_file_3',
              type=click.STRING,
              default='',
              help='path to the numpy file of test image')
@click.option(
    '--output_test_eval',
    type=click.STRING,
    default='',
    help='path to save results on test case evaluated per epoch of training')
def main(train_imgs_0_np_file,
         train_masks_0_np_file,
         train_imgs_1_np_file,
         train_masks_1_np_file,
         train_imgs_2_np_file,
         train_masks_2_np_file,
         output,
         pretrained_model='',
         pretrained_adversarial_model='',
         use_augmentation=False,
         use_weighted_crossentropy=False,
         use_cnn=False,
         use_patch_gan=False,
         use_cnn_discrimator=False,
         use_patch_gan_discrimator=False,
         test=False,
         test_imgs_np_file_1='',
         test_masks_np_file_1='',
         test_imgs_np_file_2='',
         test_masks_np_file_2='',
         test_imgs_np_file_3='',
         output_test_eval=''):
    assert (test_imgs_np_file_1 != '' and test_masks_np_file_1 != '') or (
        test_imgs_np_file_1 == '' and test_masks_np_file_1 == ''
    ), 'Both test image file and test mask file must be given'

    num_classes = 4
    if not use_augmentation:
        total_epochs = 1000
        generator_epochs = 100
    else:
        total_epochs = 1000
        generator_epochs = 100
    n_images_per_epoch = 616
    batch_size = 1
    learn_rate = 2e-4
    learn_decay = 1e-8

    eval_per_epoch = (test_imgs_np_file_1 != '' and test_masks_np_file_1 != '')
    if eval_per_epoch:
        test_imgs_1 = np.load(test_imgs_np_file_1)
        test_masks_1 = np.load(test_masks_np_file_1)
        test_imgs_2 = np.load(test_imgs_np_file_2)
        print("test_imgs_2:",test_imgs_2.shape)
        test_masks_2 = np.load(test_masks_np_file_2)
        test_imgs = np.load(test_imgs_np_file_3)
        print("test_imgs_3:",test_imgs.shape)
        test_imgs_3 = test_imgs[:15,:,:,np.newaxis]
        print("test_imgs_3_slice:",test_imgs_3.shape)
        test_imgs_4 = test_imgs[16:31,:,:,np.newaxis]
        print("test_imgs_4_slice:",test_imgs_4.shape)
        

    train_imgs = np.load(train_imgs_0_np_file)
    train_masks = np.load(train_masks_0_np_file)
    print("load_img:",train_imgs.shape)
    print("load_mask:",train_masks.shape)


    train_imgs_add = np.load(train_imgs_1_np_file)
    train_masks_add = np.load(train_masks_1_np_file)
    print("load_img:",train_imgs_add.shape)
    print("load_mask:",train_masks_add.shape)
    train_imgs = np.concatenate((train_imgs, train_imgs_add[:,:,:,np.newaxis]), axis=0)
    train_masks = np.concatenate((train_masks, train_masks_add[:,:,:,np.newaxis]), axis=0)


    train_imgs_add = np.load(train_imgs_2_np_file)
    train_masks_add = np.load(train_masks_2_np_file)
    print("load_img:",train_imgs_add.shape)
    print("load_mask:",train_masks_add.shape)
    train_imgs = np.concatenate((train_imgs, train_imgs_add[:,:,:,np.newaxis]), axis=0)
    train_masks = np.concatenate((train_masks, train_masks_add[:,:,:,np.newaxis]), axis=0)

    train_imgs_new  = train_imgs.copy()
    train_masks_new  = train_masks.copy()
    count_i = 0
    count = 0
    count_list = []
    for i in train_masks:
        count_j = 0
        sum = 0
        for j in i:
            count_j +=1
            count_k = 0
            for k in j:
                count_k +=1
                sum +=int(k[0])
        if sum == 0:
            delete_number = count_i - count
            train_imgs_new = np.delete(train_imgs_new, delete_number, axis=0)
            train_masks_new = np.delete(train_masks_new, delete_number, axis=0)

            count += 1
            print("empty:",count, count_i)
            
        count_i +=1
    train_imgs = train_imgs_new
    train_masks = train_masks_new
    print("train_imgs_result:",train_imgs.shape)
    print("train_masks_result:",train_masks.shape)

    if use_weighted_crossentropy:
        class_weights = class_weight.compute_class_weight(
            'balanced', np.unique(train_masks), train_masks.flatten())

    channels_num = train_imgs.shape[-1]
    img_shape = (train_imgs.shape[1], train_imgs.shape[2], channels_num)
    mask_shape = (train_masks.shape[1], train_masks.shape[2], num_classes)

    generator_nn = get_model(img_shape=img_shape, num_classes=num_classes)

    if pretrained_model != '':
        assert os.path.isfile(pretrained_model)
        generator_nn.load_weights(pretrained_model)

    if use_augmentation:
        samples_num = train_imgs.shape[0]
        images_aug = np.zeros(train_imgs.shape, dtype=np.float32)
        masks_aug = np.zeros(train_masks.shape, dtype=np.float32)
        for i in range(samples_num):
            images_aug[i], masks_aug[i] = augmentation(train_imgs[i],
                                                       train_masks[i])

        train_imgs = np.concatenate((train_imgs, images_aug), axis=0)
        train_masks = np.concatenate((train_masks, masks_aug), axis=0)
    print("train_imgs, train_masks, num_classes:", train_imgs.shape,
          train_masks.shape, num_classes)
    
    if use_patch_gan_discrimator:
        train_masks_cat = to_categorical(train_masks, num_classes)
        print("train_masks_cat:", train_masks_cat.shape)

    else:
        train_masks_cat = to_categorical(train_masks, num_classes)
        num_classes = 4
        print("train_masks_cat:", train_masks_cat.shape)
    
    if use_weighted_crossentropy:
        opt_discriminator = Adam(lr=(learn_rate))
        loss_function = weighted_categorical_crossentropy(class_weights)
    if use_patch_gan or use_patch_gan_discrimator:
        opt_discriminator = Adam(lr=learn_rate,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-08)
        opt_dcgan = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        loss_function = 'categorical_crossentropy'
    elif use_cnn or use_cnn_discrimator:
        opt_discriminator = Adam(lr=(learn_rate))
        loss_function = 'categorical_crossentropy'
        #loss_function = dice_coef_loss
    logging.basicConfig(filename='UNET_loss.log', level=logging.INFO)

    #use cnn discrimator
    if use_cnn_discrimator:
        print("---------use_cnn---------")
        mask_shape_discrimator = (train_masks.shape[1], train_masks.shape[2],
                                  num_classes + 1)
        optimizer = RMSprop(lr=learn_rate, clipvalue=1.0, decay=learn_decay)
        discriminator = build_discriminator(mask_shape_discrimator)
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=opt_discriminator)
        discriminator.summary()

        generator_nn.compile(loss=loss_function,
                             optimizer=opt_discriminator)
        img = Input(shape=img_shape)
        rec_mask = generator_nn(img)
        rec_mask_new = Concatenate()([rec_mask, img])
        discriminator.trainable = False
        validity = discriminator(rec_mask_new)
        adversarial_model = Model(img, [rec_mask, validity], name='D')
        
        adversarial_model.compile(
            loss=[loss_function, 'binary_crossentropy'],
            loss_weights=[1, 1],
            optimizer=optimizer)

        adversarial_model.summary()
        if pretrained_adversarial_model != '':
            assert os.path.isfile(pretrained_model)
            adversarial_model.load_weights(pretrained_model)

        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))

        current_epoch = 1
        orig_num = 16
        test_img_num = 15
        orig_rows = 480
        orig_cols = 480
        orig_rows_new = 512
        orig_cols_new = 512
        orig_mask_1 = np.zeros([orig_num, orig_rows, orig_cols],
                               dtype='float32')
        orig_mask_2 = np.zeros([orig_num, orig_rows, orig_cols],
                               dtype='float32')
        orig_mask_3 = np.zeros([test_img_num, orig_rows_new, orig_cols_new],
                               dtype='float32')
        orig_mask_4 = np.zeros([test_img_num, orig_rows, orig_cols],
                               dtype='float32')
        history_1 = {}
        history_1['dsc'] = []
        history_1['h95'] = []
        history_2 = {}
        history_2['dsc'] = []
        history_2['h95'] = []

        while current_epoch <= generator_epochs:
            print('Epoch', str(current_epoch), '/', str(generator_epochs))
            generator_nn.fit(train_imgs,
                             train_masks_cat,
                             batch_size=batch_size,
                             epochs=1,
                             verbose=True,
                             shuffle=True)
            current_epoch += 1 

        current_epoch = 1
        while current_epoch <= total_epochs:
            print('Epoch', str(current_epoch), '/', str(total_epochs))
            output_weights_file = output + '/' + 'weight_' + str(
                current_epoch) + '.h5'
            batch_idxs = len(train_imgs) // batch_size
            progbar = keras_generic_utils.Progbar(batch_idxs)

            for idx in range(0, batch_idxs):
                img_batch = train_imgs[idx * batch_size:(idx + 1) * batch_size]
                masks_cat_batch = train_masks_cat[idx * batch_size:(idx + 1) *
                                                  batch_size]
                masks_cat_batch_new = np.concatenate((masks_cat_batch, img_batch),
                                           axis=3)
                recon_masks_cat = generator_nn.predict(img_batch)
                recon_masks_cat_new = np.concatenate((recon_masks_cat, img_batch),
                                           axis=3)
                d_loss_real = discriminator.train_on_batch(
                    masks_cat_batch_new, ones)
                d_loss_fake = discriminator.train_on_batch(
                    recon_masks_cat_new, zeros)
                adversarial_model.train_on_batch(img_batch,
                                                 [masks_cat_batch, ones])

                g_loss = adversarial_model.train_on_batch(
                    img_batch, [masks_cat_batch, ones])

                D_log_loss = d_loss_real + d_loss_fake
                gen_total_loss = g_loss[0].tolist()
                gen_total_loss = min(gen_total_loss, 1000000)
                gen_mae = g_loss[1].tolist()
                gen_mae = min(gen_mae, 1000000)
                gen_log_loss = g_loss[2].tolist()
                gen_log_loss = min(gen_log_loss, 1000000)

                progbar.add(batch_size,
                            values=[("Dis logloss", D_log_loss),
                                    ("Gen total", gen_total_loss),
                                    ("Gen L1 (mae)", gen_mae),
                                    ("Gen logloss", gen_log_loss)])
             # ------------------------------
            # save weights on every 2nd epoch
            if current_epoch % 5 == 0:
                gen_weights_path = os.path.join(
                    './weight/gen_weights_epoch_%s.h5' % (current_epoch))
                generator_nn.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join(
                    './weight/disc_weights_epoch_%s.h5' % (current_epoch))
                discriminator.save_weights(disc_weights_path, overwrite=True)

                adversarial_weights_path = os.path.join(
                    './weight/arg_adversarial_weights_epoch_%s.h5' % (current_epoch))
                adversarial_model.save_weights(adversarial_weights_path, overwrite=True)
            if eval_per_epoch and current_epoch % 100 == 0:
                pred_masks_1 = adversarial_model.predict(test_imgs_1)
                pred_masks_1 = pred_masks_1[0].argmax(axis=3)
                pred_masks_2 = adversarial_model.predict(test_imgs_2)
                pred_masks_2 = pred_masks_2[0].argmax(axis=3)
                pred_masks_3 = adversarial_model.predict(test_imgs_3)
                pred_masks_3 = pred_masks_3[0].argmax(axis=3)
                pred_masks_4 = adversarial_model.predict(test_imgs_4)
                pred_masks_4 = pred_masks_4[0].argmax(axis=3)
                rows = np.shape(pred_masks_1)[1]
                cols = np.shape(pred_masks_1)[2]
                orig_mask_1[:,
                            int((orig_rows - rows) /
                                2):int((orig_rows - rows) / 2) + rows,
                            int((orig_cols - cols) /
                                2):int((orig_cols - cols) / 2) +
                            cols] = pred_masks_1
                orig_mask_2[:,
                            int((orig_rows - rows) /
                                2):int((orig_rows - rows) / 2) + rows,
                            int((orig_cols - cols) /
                                2):int((orig_cols - cols) / 2) +
                            cols] = pred_masks_2
                orig_mask_3[:,
                            int((orig_rows_new - rows) /
                                2):int((orig_rows_new - rows) / 2) + rows,
                            int((orig_rows_new - cols) /
                                2):int((orig_rows_new - cols) / 2) +
                            cols] = pred_masks_3
                orig_mask_4[:,
                            int((orig_rows - rows) /
                                2):int((orig_rows - rows) / 2) + rows,
                            int((orig_rows - cols) /
                                2):int((orig_rows - cols) / 2) +
                            cols] = pred_masks_4
                
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_1),
                    output + '/' + test_imgs_np_file_1[0:6] +
                    '_4_cnn_pred_epoch_' + str(current_epoch) + '.nii.gz')
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_2),
                    output + '/' + test_imgs_np_file_2[0:6] +
                    '_5_cnn_pred_epoch_' + str(current_epoch) + '.nii.gz')
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_3),
                    output + '/' + test_imgs_np_file_3[0:6] +
                    '_6_cnn_pred_epoch_' + str(current_epoch) + '.nii.gz')
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_4),
                    output + '/' + test_imgs_np_file_3[0:6] +
                    '_7_cnn_pred_epoch_' + str(current_epoch) + '.nii.gz')

                dsc, h95, vs = get_eval_metrics(test_masks_1[:, :, :, 0],
                                                pred_masks_1)
                history_1['dsc'].append(dsc)
                history_1['h95'].append(h95)
                print(dsc)
                print(h95)
                dsc, h95, vs = get_eval_metrics(test_masks_2[:, :, :, 0],
                                                pred_masks_2)
                history_2['dsc'].append(dsc)
                history_2['h95'].append(h95)
                print(dsc)
                print(h95)

            
                if output_test_eval != '':
                    with open(output_test_eval, 'w+') as outfile:
                        json.dump(history_1, outfile)
                        json.dump(history_2, outfile)

            current_epoch += 1

        if output_test_eval != '':
            with open(output_test_eval, 'w+') as outfile:
                json.dump(history_1, outfile)
                json.dump(history_2, outfile)

    # use patch gan discrimator
    if use_patch_gan_discrimator:
        sub_patch_dim = (28, 28)
        mask_shape_discrimator = (train_masks.shape[1], train_masks.shape[2],
                                  num_classes + 1)
        nb_patch_patches, patch_gan_dim = patch_utils.num_patches(
            output_img_dim=mask_shape_discrimator, sub_patch_dim=sub_patch_dim)
        print("nb_patch_patches:", nb_patch_patches)
        print("patch_gan_dim:", patch_gan_dim)
        discriminator = PatchGanDiscriminator(
            output_img_dim=mask_shape_discrimator,
            patch_dim=patch_gan_dim,
            nb_patches=nb_patch_patches)
        discriminator.summary()
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=opt_discriminator)
        # disable training while we put it through the GAN
        discriminator.trainable = False

        # compile generator
        generator_nn.compile(loss=loss_function, optimizer=opt_discriminator)

        dc_gan_nn = DCGAN(generator_model=generator_nn,
                          discriminator_model=discriminator,
                          input_img_dim=img_shape,
                          patch_dim=sub_patch_dim,
                          use_patch_gan_discrimator=use_patch_gan_discrimator)
        dc_gan_nn.summary()
        # Compile DCGAN
        loss = [loss_function, 'binary_crossentropy']
        loss_weights = [1, 1]
        dc_gan_nn.compile(loss=loss,
                          loss_weights=loss_weights,
                          optimizer=opt_dcgan)

        # ENABLE DISCRIMINATOR AND COMPILE
        

        print('Training starting...')

        current_epoch = 1
        orig_num = 16
        orig_rows = 480
        orig_cols = 480
        orig_mask_1 = np.zeros([orig_num, orig_rows, orig_cols],
                               dtype='float32')
        orig_mask_2 = np.zeros([orig_num, orig_rows, orig_cols],
                               dtype='float32')
        history_1 = {}
        history_1['dsc'] = []
        history_1['h95'] = []
        history_2 = {}
        history_2['dsc'] = []
        history_2['h95'] = []
        while current_epoch <= generator_epochs:
            print('Epoch', str(current_epoch), '/', str(generator_epochs))
            generator_nn.fit(train_imgs,
                             train_masks_cat,
                             batch_size=batch_size,
                             epochs=1,
                             verbose=True,
                             shuffle=True)
            current_epoch += 1 

        for epoch in range(0, total_epochs):

            print('Epoch {}'.format(epoch))

            start = time.time()
            progbar = keras_generic_utils.Progbar(n_images_per_epoch)
            # init the datasources again for each epoch
            batch_idxs = len(train_imgs) // batch_size
            progbar = keras_generic_utils.Progbar(batch_idxs)
            for idx in range(0, batch_idxs):
                img_batch = train_imgs[idx * batch_size:(idx + 1) * batch_size]
                masks_cat_batch = train_masks_cat[idx * batch_size:(idx + 1) *
                                                  batch_size]

                mask_disc = np.concatenate((masks_cat_batch, img_batch),
                                           axis=3)

                print(" ")
                mask_disc = patch_utils.extract_patches(
                    images=mask_disc, sub_patch_dim=sub_patch_dim)

                gen_mask_disc = generator_nn.predict(img_batch)
                gen_mask_disc = np.concatenate((gen_mask_disc, img_batch),
                                               axis=3)

                gen_mask_disc = patch_utils.extract_patches(
                    images=gen_mask_disc, sub_patch_dim=sub_patch_dim)

                ones = np.ones((batch_size, 1))
                zeros = np.zeros((batch_size, 1))
                # Update the discriminator
                d_loss_real = discriminator.train_on_batch(mask_disc, ones)
                d_loss_fake = discriminator.train_on_batch(
                    gen_mask_disc, zeros)

                # trainining GAN
                # print('calculating GAN loss...')
                gen_loss = dc_gan_nn.train_on_batch(img_batch,
                                                    [masks_cat_batch, ones])

                # Unfreeze the discriminator
                # discriminator.trainable = True
                # counts batches we've ran through for generating fake vs real images

                # print losses
                D_log_loss = d_loss_real + d_loss_fake
                gen_total_loss = gen_loss[0].tolist()
                gen_total_loss = min(gen_total_loss, 1000000)
                gen_mae = gen_loss[1].tolist()
                gen_mae = min(gen_mae, 1000000)
                gen_log_loss = gen_loss[2].tolist()
                gen_log_loss = min(gen_log_loss, 1000000)

                progbar.add(batch_size,
                            values=[("Dis logloss", D_log_loss),
                                    ("Gen total", gen_total_loss),
                                    ("Gen L1 (mae)", gen_mae),
                                    ("Gen logloss", gen_log_loss)])

                # ---------------------------
                # Save images for visualization every 2nd batch
            if eval_per_epoch and epoch % 10 == 0:
                pred_masks_1 = dc_gan_nn.predict(test_imgs_1)
                pred_masks_1 = pred_masks_1[0].argmax(axis=3)
                pred_masks_2 = dc_gan_nn.predict(test_imgs_2)
                pred_masks_2 = pred_masks_2[0].argmax(axis=3)
                rows = np.shape(pred_masks_1)[1]
                cols = np.shape(pred_masks_1)[2]
                orig_mask_1[:,
                            int((orig_rows - rows) /
                                2):int((orig_rows - rows) / 2) + rows,
                            int((orig_cols - cols) /
                                2):int((orig_cols - cols) / 2) +
                            cols] = pred_masks_1
                orig_mask_2[:,
                            int((orig_rows - rows) /
                                2):int((orig_rows - rows) / 2) + rows,
                            int((orig_cols - cols) /
                                2):int((orig_cols - cols) / 2) +
                            cols] = pred_masks_2
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_1),
                    output + '/' + test_imgs_np_file_1[0:6] +
                    '_patch_gan_pred_epoch_' + str(epoch) + '.nii.gz')
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_2),
                    output + '/' + test_imgs_np_file_2[0:6] +
                    '_patch_gan_pred_epoch_' + str(epoch) + '.nii.gz')

                dsc, h95, vs = get_eval_metrics(test_masks_1[:, :, :, 0],
                                                pred_masks_1)
                history_1['dsc'].append(dsc)
                history_1['h95'].append(h95)
                print(dsc)
                print(h95)
                dsc, h95, vs = get_eval_metrics(test_masks_2[:, :, :, 0],
                                                pred_masks_2)
                history_2['dsc'].append(dsc)
                history_2['h95'].append(h95)
                print(dsc)
                print(h95)

                if output_test_eval != '':
                    with open(output_test_eval, 'w+') as outfile:
                        json.dump(history_1, outfile)
                        json.dump(history_2, outfile)

            

            # -----------------------
            # print losses
                D_log_loss = d_loss_real + d_loss_fake
                gen_total_loss = g_loss[0].tolist()
                gen_total_loss = min(gen_total_loss, 1000000)
                gen_mae = g_loss[1].tolist()
                gen_mae = min(gen_mae, 1000000)
                gen_log_loss = g_loss[2].tolist()
                gen_log_loss = min(gen_log_loss, 1000000)

                progbar.add(batch_size,
                            values=[("Dis logloss", D_log_loss),
                                    ("Gen total", gen_total_loss),
                                    ("Gen L1 (mae)", gen_mae),
                                    ("Gen logloss", gen_log_loss)])
            if eval_per_epoch and epoch % 1 == 0:
                pred_masks_1 = dc_gan_nn.predict(test_imgs_1)
                pred_masks_1 = pred_masks_1[0].argmax(axis=3)
                pred_masks_2 = dc_gan_nn.predict(test_imgs_2)
                pred_masks_2 = pred_masks_2[0].argmax(axis=3)
                rows = np.shape(pred_masks_1)[1]
                cols = np.shape(pred_masks_1)[2]
                orig_mask_1[:,
                            int((orig_rows - rows) /
                                2):int((orig_rows - rows) / 2) + rows,
                            int((orig_cols - cols) /
                                2):int((orig_cols - cols) / 2) +
                            cols] = pred_masks_1
                orig_mask_2[:,
                            int((orig_rows - rows) /
                                2):int((orig_rows - rows) / 2) + rows,
                            int((orig_cols - cols) /
                                2):int((orig_cols - cols) / 2) +
                            cols] = pred_masks_2
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_1),
                    output + '/' + test_imgs_np_file_1[0:6] +
                    '_patch_gan_pred_epoch_' + str(current_epoch) + '.nii.gz')
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_2),
                    output + '/' + test_imgs_np_file_2[0:6] +
                    '_patch_gan_pred_epoch_' + str(current_epoch) + '.nii.gz')

                dsc, h95, vs = get_eval_metrics(test_masks_1[:, :, :, 0],
                                                pred_masks_1)
                history_1['dsc'].append(dsc)
                history_1['h95'].append(h95)
                print(dsc)
                print(h95)
                dsc, h95, vs = get_eval_metrics(test_masks_2[:, :, :, 0],
                                                pred_masks_2)
                history_2['dsc'].append(dsc)
                history_2['h95'].append(h95)
                print(dsc)
                print(h95)

                if output_test_eval != '':
                    with open(output_test_eval, 'w+') as outfile:
                        json.dump(history_1, outfile)
                        json.dump(history_2, outfile)
            current_epoch += 1
            # ------------------------------
            # save weights on every 2nd epoch
            if epoch % 2 == 0:
                gen_weights_path = os.path.join(
                    './weight/gen_weights_epoch_%s.h5' % (epoch))
                generator_nn.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join(
                    './weight/disc_weights_epoch_%s.h5' % (epoch))
                discriminator.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join(
                    './weight/DCGAN_weights_epoch_%s.h5' % (epoch))
                dc_gan_nn.save_weights(DCGAN_weights_path, overwrite=True)

        if output_test_eval != '':
            with open(output_test_eval, 'w+') as outfile:
                json.dump(history_1, outfile)
                json.dump(history_2, outfile)

    # orignial patch GAN model
    if use_patch_gan:
        sub_patch_dim = (56, 56)
        nb_patch_patches, patch_gan_dim = patch_utils.num_patches(
            output_img_dim=mask_shape, sub_patch_dim=sub_patch_dim)
        print("nb_patch_patches:", nb_patch_patches)
        print("patch_gan_dim:", patch_gan_dim)
        discriminator = PatchGanDiscriminator(output_img_dim=mask_shape,
                                              patch_dim=patch_gan_dim,
                                              nb_patches=nb_patch_patches)
        discriminator.summary()
        # disable training while we put it through the GAN
        discriminator.trainable = False

        # compile generator
        generator_nn.compile(loss=loss_function, optimizer=opt_discriminator)

        dc_gan_nn = DCGAN(generator_model=generator_nn,
                          discriminator_model=discriminator,
                          input_img_dim=img_shape,
                          patch_dim=sub_patch_dim,
                          use_patch_gan_discrimator=use_patch_gan_discrimator)
        dc_gan_nn.summary()
        # Compile DCGAN
        loss = [loss_function, 'binary_crossentropy']
        loss_weights = [1E2, 1]
        dc_gan_nn.compile(loss=loss,
                          loss_weights=loss_weights,
                          optimizer=opt_dcgan)

        # ENABLE DISCRIMINATOR AND COMPILE
        discriminator.trainable = True
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=opt_discriminator)

        print('Training starting...')

        current_epoch = 1
        orig_num = 3
        orig_rows = 480
        orig_cols = 480
        orig_mask_1 = np.zeros([orig_num, orig_rows, orig_cols],
                               dtype='float32')
        orig_mask_2 = np.zeros([orig_num, orig_rows, orig_cols],
                               dtype='float32')
        history_1 = {}
        history_1['dsc'] = []
        history_1['h95'] = []
        history_2 = {}
        history_2['dsc'] = []
        history_2['h95'] = []

        for epoch in range(0, total_epochs):

            print('Epoch {}'.format(epoch))

            start = time.time()
            progbar = keras_generic_utils.Progbar(n_images_per_epoch)

            # init the datasources again for each epoch

            batch_idxs = len(train_imgs) // batch_size
            for idx in range(0, batch_idxs):
                img_batch = train_imgs[idx * batch_size:(idx + 1) * batch_size]
                masks_cat_batch = train_masks_cat[idx * batch_size:(idx + 1) *
                                                  batch_size]

                mask_disc = masks_cat_batch
                mask_disc = patch_utils.extract_patches(
                    images=mask_disc, sub_patch_dim=sub_patch_dim)
                # patch_utils.get_disc_batch
                gen_mask_disc = generator_nn.predict(img_batch)
                # print("gen_mask_disc:",gen_mask_disc.shape)
                gen_mask_disc = patch_utils.extract_patches(
                    images=gen_mask_disc, sub_patch_dim=sub_patch_dim)

                ones = np.ones((batch_size, 1))
                zeros = np.zeros((batch_size, 1))
                # Update the discriminator
                d_real_loss = discriminator.train_on_batch(mask_disc, ones)
                d_fake_loss = discriminator.train_on_batch(
                    gen_mask_disc, zeros)
                print("d_real_loss, d_fake_loss:", d_real_loss, d_fake_loss)
                # Freeze the discriminator
                discriminator.trainable = False

                # trainining GAN
                # print('calculating GAN loss...')
                gen_loss = dc_gan_nn.train_on_batch(img_batch,
                                                    [masks_cat_batch, ones])

                # Unfreeze the discriminator
                discriminator.trainable = True

                # print losses
                D_log_loss = d_real_loss + d_fake_loss
                gen_total_loss = gen_loss[0].tolist()
                gen_total_loss = min(gen_total_loss, 1000000)
                gen_mae = gen_loss[1].tolist()
                gen_mae = min(gen_mae, 1000000)
                gen_log_loss = gen_loss[2].tolist()
                gen_log_loss = min(gen_log_loss, 1000000)

                progbar.add(batch_size,
                            values=[("Dis logloss", D_log_loss),
                                    ("Gen total", gen_total_loss),
                                    ("Gen L1 (mae)", gen_mae),
                                    ("Gen logloss", gen_log_loss)])

                # ---------------------------
                # Save images for visualization every 2nd batch
            if eval_per_epoch and epoch % 10 == 0:
                pred_masks_1 = dc_gan_nn.predict(test_imgs_1)
                pred_masks_1 = pred_masks_1[0].argmax(axis=3)
                pred_masks_2 = dc_gan_nn.predict(test_imgs_2)
                pred_masks_2 = pred_masks_2[0].argmax(axis=3)
                rows = np.shape(pred_masks_1)[1]
                cols = np.shape(pred_masks_1)[2]
                orig_mask_1[:,
                            int((orig_rows - rows) /
                                2):int((orig_rows - rows) / 2) + rows,
                            int((orig_cols - cols) /
                                2):int((orig_cols - cols) / 2) +
                            cols] = pred_masks_1
                orig_mask_2[:,
                            int((orig_rows - rows) /
                                2):int((orig_rows - rows) / 2) + rows,
                            int((orig_cols - cols) /
                                2):int((orig_cols - cols) / 2) +
                            cols] = pred_masks_2
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_1),
                    output + '/' + test_imgs_np_file_1[0:6] +
                    '_patch_gan_pred_epoch_' + str(current_epoch) + '.nii.gz')
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_2),
                    output + '/' + test_imgs_np_file_2[0:6] +
                    '_patch_gan_pred_epoch_' + str(current_epoch) + '.nii.gz')

                dsc, h95, vs = get_eval_metrics(test_masks_1[:, :, :, 0],
                                                pred_masks_1)
                history_1['dsc'].append(dsc)
                history_1['h95'].append(h95)
                print(dsc)
                print(h95)
                dsc, h95, vs = get_eval_metrics(test_masks_2[:, :, :, 0],
                                                pred_masks_2)
                history_2['dsc'].append(dsc)
                history_2['h95'].append(h95)
                print(dsc)
                print(h95)

                if output_test_eval != '':
                    with open(output_test_eval, 'w+') as outfile:
                        json.dump(history_1, outfile)
                        json.dump(history_2, outfile)

            current_epoch += 1

            # -----------------------
            # log epoch
            print("")
            print('Epoch %s/%s, Time: %s' %
                  (epoch + 1, total_epochs, time.time() - start))

            # ------------------------------
            # save weights on every 2nd epoch
            if epoch % 2 == 0:
                gen_weights_path = os.path.join(
                    './weight/gen_weights_epoch_%s.h5' % (epoch))
                generator_nn.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join(
                    './weight/disc_weights_epoch_%s.h5' % (epoch))
                discriminator.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join(
                    './weight/DCGAN_weights_epoch_%s.h5' % (epoch))
                dc_gan_nn.save_weights(DCGAN_weights_path, overwrite=True)

        if output_test_eval != '':
            with open(output_test_eval, 'w+') as outfile:
                json.dump(history_1, outfile)
                json.dump(history_2, outfile)

    # original CNN model
    elif use_cnn:
        print("---------use_cnn---------")
        optimizer = RMSprop(lr=learn_rate, clipvalue=1.0, decay=learn_decay)
        discriminator = build_discriminator(mask_shape)
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=opt_discriminator)
        discriminator.summary()
        generator_nn.compile(loss=loss_function,
                             optimizer=opt_discriminator)
        img = Input(shape=img_shape)
        rec_mask = generator_nn(img)

        discriminator.trainable = False
        validity = discriminator(rec_mask)
        adversarial_model = Model(img, [rec_mask, validity], name='D')
        adversarial_model.compile(
            loss=[loss_function, 'binary_crossentropy'],
            loss_weights=[1, 1],
            optimizer=optimizer)
        # discriminator.trainable = False
        # discriminator.compile(loss='binary_crossentropy',
        #                       optimizer=opt_discriminator)
        adversarial_model.summary()
        if pretrained_adversarial_model != '':
            assert os.path.isfile(pretrained_model)
            adversarial_model.load_weights(pretrained_model)

        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))

        current_epoch = 1
        orig_num = 3
        orig_rows = 480
        orig_cols = 480
        orig_mask_1 = np.zeros([orig_num, orig_rows, orig_cols],
                               dtype='float32')
        orig_mask_2 = np.zeros([orig_num, orig_rows, orig_cols],
                               dtype='float32')
        history_1 = {}
        history_1['dsc'] = []
        history_1['h95'] = []
        history_2 = {}
        history_2['dsc'] = []
        history_2['h95'] = []

        while current_epoch <= total_epochs:
            print('Epoch', str(current_epoch), '/', str(total_epochs))
            output_weights_file = output + '/' + 'weight_' + str(
                current_epoch) + '.h5'
            batch_idxs = len(train_imgs) // batch_size
            for idx in range(0, batch_idxs):
                img_batch = train_imgs[idx * batch_size:(idx + 1) * batch_size]
                masks_cat_batch = train_masks_cat[idx * batch_size:(idx + 1) *
                                                  batch_size]

                recon_masks_cat = generator_nn.predict(img_batch)

                d_loss_real = discriminator.train_on_batch(
                    masks_cat_batch, ones)
                d_loss_fake = discriminator.train_on_batch(
                    recon_masks_cat, zeros)
                # discriminator.trainable = False
                adversarial_model.train_on_batch(img_batch,
                                                 [masks_cat_batch, ones])

                g_loss = adversarial_model.train_on_batch(
                    img_batch, [masks_cat_batch, ones])

                # discriminator.trainable = True
                msg = 'Epoch:[{0}]-[{1}/{2}] --> d_loss: {3:>0.3f}'.format( \
                                current_epoch, idx, batch_idxs, d_loss_real+d_loss_fake)+" "+str(g_loss[0]) +" "+ str(g_loss[1])
                print(msg)
                logging.info(msg)
            if eval_per_epoch and current_epoch % 10 == 0:
                pred_masks_1 = adversarial_model.predict(test_imgs_1)
                pred_masks_1 = pred_masks_1[0].argmax(axis=3)
                pred_masks_2 = adversarial_model.predict(test_imgs_2)
                pred_masks_2 = pred_masks_2[0].argmax(axis=3)
                rows = np.shape(pred_masks_1)[1]
                cols = np.shape(pred_masks_1)[2]
                orig_mask_1[:,
                            int((orig_rows - rows) /
                                2):int((orig_rows - rows) / 2) + rows,
                            int((orig_cols - cols) /
                                2):int((orig_cols - cols) / 2) +
                            cols] = pred_masks_1
                orig_mask_2[:,
                            int((orig_rows - rows) /
                                2):int((orig_rows - rows) / 2) + rows,
                            int((orig_cols - cols) /
                                2):int((orig_cols - cols) / 2) +
                            cols] = pred_masks_2
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_1),
                    output + '/' + test_imgs_np_file_1[0:6] +
                    '_cnn_pred_epoch_' + str(current_epoch) + '.nii.gz')
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_2),
                    output + '/' + test_imgs_np_file_2[0:6] +
                    '_cnn_pred_epoch_' + str(current_epoch) + '.nii.gz')

                dsc, h95, vs = get_eval_metrics(test_masks_1[:, :, :, 0],
                                                pred_masks_1)
                history_1['dsc'].append(dsc)
                history_1['h95'].append(h95)
                print(dsc)
                print(h95)
                dsc, h95, vs = get_eval_metrics(test_masks_2[:, :, :, 0],
                                                pred_masks_2)
                history_2['dsc'].append(dsc)
                history_2['h95'].append(h95)
                print(dsc)
                print(h95)

                if output_test_eval != '':
                    with open(output_test_eval, 'w+') as outfile:
                        json.dump(history_1, outfile)
                        json.dump(history_2, outfile)

            current_epoch += 1

        if output_test_eval != '':
            with open(output_test_eval, 'w+') as outfile:
                json.dump(history_1, outfile)
                json.dump(history_2, outfile)

    elif test:
        optimizer = RMSprop(lr=learn_rate, clipvalue=1.0, decay=learn_decay)

        img = Input(shape=img_shape)
        rec_mask = generator_nn(img)
        discriminator = build_discriminator(mask_shape)
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=opt_discriminator,
                              metrics=['accuracy'])
        generator_img = Model(img, rec_mask, name='D')
        discriminator.trainable = False

        valid = discriminator(rec_mask)
        generator_val = Model(img, valid, name='V')

        discriminator.summary()
        generator_val.summary()
        generator_img.summary()

        generator_img.compile(loss=loss_function,
                              optimizer=optimizer)

        generator_val.compile(loss='binary_crossentropy', optimizer=optimizer)

        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))

        current_epoch = 1
        orig_num = 3
        orig_rows = 480
        orig_cols = 480
        orig_mask_1 = np.zeros([orig_num, orig_rows, orig_cols],
                               dtype='float32')
        orig_mask_2 = np.zeros([orig_num, orig_rows, orig_cols],
                               dtype='float32')
        history_1 = {}
        history_1['dsc'] = []
        history_1['h95'] = []
        history_2 = {}
        history_2['dsc'] = []
        history_2['h95'] = []

        while current_epoch <= total_epochs:
            print('Epoch', str(current_epoch), '/', str(total_epochs))
            output_weights_file = output + '/' + 'weight_' + str(
                current_epoch) + '.h5'
            batch_idxs = len(train_imgs) // batch_size
            for idx in range(0, batch_idxs):
                img_batch = train_imgs[idx * batch_size:(idx + 1) * batch_size]
                masks_cat_batch = train_masks_cat[idx * batch_size:(idx + 1) *
                                                  batch_size]

                recon_masks_cat = generator_nn.predict(img_batch)

                d_loss_real = discriminator.train_on_batch(
                    masks_cat_batch, ones)
                d_loss_fake = discriminator.train_on_batch(
                    recon_masks_cat, zeros)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                g_loss_img = generator_img.train_on_batch(
                    img_batch, masks_cat_batch)
                g_loss_val = generator_val.train_on_batch(img_batch, ones)

                msg = 'Epoch:[{0}]-[{1}/{2}] --> d_loss:'.format( \
                                current_epoch, idx, batch_idxs)
                msg = msg + " " + str(d_loss) + " " + str(
                    g_loss_val) + " " + str(g_loss_img)
                print(msg)
                logging.info(msg)
            if eval_per_epoch and current_epoch % 10 == 0:
                pred_masks_1 = generator_nn.predict(test_imgs_1)
                pred_masks_1 = pred_masks_1.argmax(axis=3)
                pred_masks_2 = generator_nn.predict(test_imgs_2)
                pred_masks_2 = pred_masks_2.argmax(axis=3)
                rows = np.shape(pred_masks_1)[1]
                cols = np.shape(pred_masks_1)[2]
                orig_mask_1[:,
                            int((orig_rows - rows) /
                                2):int((orig_rows - rows) / 2) + rows,
                            int((orig_cols - cols) /
                                2):int((orig_cols - cols) / 2) +
                            cols] = pred_masks_1
                orig_mask_2[:,
                            int((orig_rows - rows) /
                                2):int((orig_rows - rows) / 2) + rows,
                            int((orig_cols - cols) /
                                2):int((orig_cols - cols) / 2) +
                            cols] = pred_masks_2
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_1),
                    output + '/' + test_imgs_np_file_1[0:6] + '_pred_epoch_' +
                    str(current_epoch) + '.nii.gz')
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_2),
                    output + '/' + test_imgs_np_file_2[0:6] + '_pred_epoch_' +
                    str(current_epoch) + '.nii.gz')

                dsc, h95, vs = get_eval_metrics(test_masks_1[:, :, :, 0],
                                                pred_masks_1)
                history_1['dsc'].append(dsc)
                history_1['h95'].append(h95)
                print(dsc)
                print(h95)
                dsc, h95, vs = get_eval_metrics(test_masks_2[:, :, :, 0],
                                                pred_masks_2)
                history_2['dsc'].append(dsc)
                history_2['h95'].append(h95)
                print(dsc)
                print(h95)

                if output_test_eval != '':
                    with open(output_test_eval, 'w+') as outfile:
                        json.dump(history_1, outfile)
                        json.dump(history_2, outfile)

            current_epoch += 1

        if output_test_eval != '':
            with open(output_test_eval, 'w+') as outfile:
                json.dump(history_1, outfile)
                json.dump(history_2, outfile)


if __name__ == "__main__":
    main()
