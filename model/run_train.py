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
from keras.optimizers import Adam
from keras.layers import Input
from keras.utils import to_categorical
from keras.models import Model
from keras.utils import generic_utils as keras_generic_utils

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


@click.command()
@click.argument('train_imgs_np_file', type=click.STRING)
@click.argument('train_masks_np_file', type=click.STRING)
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
@click.option('--use_pretrained_unet_patch_gan',
              type=click.BOOL,
              default=False,
              help='use pretrained unet patch gan or not')
@click.option('--use_pretrained_unet_cnn',
              type=click.BOOL,
              default=False,
              help='use pretrained unet cnn or not')
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
@click.option(
    '--output_test_eval',
    type=click.STRING,
    default='',
    help='path to save results on test case evaluated per epoch of training')
def main(train_imgs_np_file,
         train_masks_np_file,
         output,
         pretrained_model='',
         pretrained_adversarial_model='',
         use_augmentation=False,
         use_weighted_crossentropy=False,
         use_patch_gan=False,
         use_pretrained_unet_patch_gan=False,
         use_pretrained_unet_cnn=False,
         use_patch_gan_discrimator=False,
         use_cnn=False,
         test=False,
         test_imgs_np_file_1='',
         test_masks_np_file_1='',
         test_imgs_np_file_2='',
         test_masks_np_file_2='',
         output_test_eval=''):
    assert (test_imgs_np_file_1 != '' and test_masks_np_file_1 != '') or (
        test_imgs_np_file_1 == '' and test_masks_np_file_1 == ''
    ), 'Both test image file and test mask file must be given'
    if use_patch_gan_discrimator:
        num_classes = 4
    else:
        num_classes = 4
    if not use_augmentation:
        total_epochs = 100

    else:
        total_epochs = 500
    n_images_per_epoch = 616
    batch_size = 16
    learn_rate = 2e-4

    eval_per_epoch = (test_imgs_np_file_1 != '' and test_masks_np_file_1 != '')
    if eval_per_epoch:
        test_imgs_1 = np.load(test_imgs_np_file_1)
        test_masks_1 = np.load(test_masks_np_file_1)
        test_imgs_2 = np.load(test_imgs_np_file_2)
        test_masks_2 = np.load(test_masks_np_file_2)

    train_imgs = np.load(train_imgs_np_file)
    print("train_imgs:", train_imgs)
    train_masks = np.load(train_masks_np_file)
    print("train_masks:", train_masks)

    if use_weighted_crossentropy:
        class_weights = class_weight.compute_class_weight(
            'balanced', np.unique(train_masks), train_masks.flatten())

    channels_num = train_imgs.shape[-1]
    print("channels_num:", channels_num)
    img_shape = (train_imgs.shape[1], train_imgs.shape[2], channels_num)
    mask_shape = (train_masks.shape[1], train_masks.shape[2], num_classes)
    print("img_shape:", img_shape)
    print("mask_shape:", mask_shape)

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
          train_masks.shape, num_classes - 1)
    if use_patch_gan_discrimator:
        train_masks_cat = to_categorical(train_masks, num_classes)
        print("train_masks_cat:", train_masks_cat.shape)

        # train_masks_cat_new = np.concatenate((train_masks_cat, train_imgs), axis=3)
        # # train_masks_cat_new = np.concatenate(train_masks_cat, train_imgs)

        # print("train_masks_cat_new:",train_masks_cat_new.shape)
        # print("train_masks_cat_new:", train_masks_cat_new.shape)
    else:
        train_masks_cat = to_categorical(train_masks, num_classes)

    if use_weighted_crossentropy:
        opt_discriminator = Adam(lr=(learn_rate))
        loss_function = weighted_categorical_crossentropy(class_weights)
    if use_patch_gan or use_pretrained_unet_patch_gan or use_patch_gan_discrimator:
        opt_discriminator = Adam(lr=1E-4,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-08)
        opt_dcgan = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        loss_function = 'categorical_crossentropy'
    elif use_cnn or use_pretrained_unet_cnn:
        opt_discriminator = Adam(lr=(learn_rate))
        loss_function = 'categorical_crossentropy'
    logging.basicConfig(filename='UNET_loss.log', level=logging.INFO)

    # patch GAN model + image discrimator
    if use_patch_gan_discrimator:
        sub_patch_dim = (56, 56)
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
            batch_counter = 1
            start = time.time()
            progbar = keras_generic_utils.Progbar(n_images_per_epoch)

            # init the datasources again for each epoch

            batch_idxs = len(train_imgs) // batch_size
            for idx in range(0, batch_idxs):
                img_batch = train_imgs[idx * batch_size:(idx + 1) * batch_size]
                masks_cat_batch = train_masks_cat[idx * batch_size:(idx + 1) *
                                                  batch_size]
                # masks_cat_batch_new = train_masks_cat_new[idx * batch_size:(idx + 1) *
                #                                   batch_size]

                mask_disc = np.concatenate((masks_cat_batch, img_batch),
                                           axis=3)
                print("-----------------------------")
                print("mask_disc:", mask_disc[0,:,:,5])
                print("-----------------------------")
                # train_masks_cat_new = np.concatenate((train_masks_cat, train_imgs), axis=3)
                mask_disc = patch_utils.extract_patches(
                    images=mask_disc, sub_patch_dim=sub_patch_dim)
                # patch_utils.get_disc_batch
                gen_mask_disc = generator_nn.predict(img_batch)
                gen_mask_disc = np.concatenate((gen_mask_disc, img_batch),
                                               axis=3)
                print("gen_mask_disc:", gen_mask_disc.shape)
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
                # counts batches we've ran through for generating fake vs real images
                batch_counter += 1

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

    # pretrained unet patch gan
    if use_pretrained_unet_patch_gan:
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
                          patch_dim=sub_patch_dim)
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
            generator_nn.fit(train_imgs,
                             train_masks_cat,
                             batch_size=batch_size,
                             epochs=1,
                             verbose=True,
                             shuffle=True)
            print('Epoch', str(epoch), '/', str(total_epochs))
        for epoch in range(0, total_epochs):

            print('Epoch {}'.format(epoch))
            batch_counter = 1
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
                # counts batches we've ran through for generating fake vs real images
                batch_counter += 1

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
                    '_pretrained_unet_patch_gan_pred_epoch_' +
                    str(current_epoch) + '.nii.gz')
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_2),
                    output + '/' + test_imgs_np_file_2[0:6] +
                    '_pretrained_unet_patch_gan_pred_epoch_' +
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
    # patch GAN model + image discrimator
    if use_patch_gan_discrimator:
        sub_patch_dim = (56, 56)
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
            batch_counter = 1
            start = time.time()
            progbar = keras_generic_utils.Progbar(n_images_per_epoch)

            # init the datasources again for each epoch

            batch_idxs = len(train_imgs) // batch_size
            for idx in range(0, batch_idxs):
                img_batch = train_imgs[idx * batch_size:(idx + 1) * batch_size]
                masks_cat_batch = train_masks_cat[idx * batch_size:(idx + 1) *
                                                  batch_size]
                # masks_cat_batch_new = train_masks_cat_new[idx * batch_size:(idx + 1) *
                #                                   batch_size]

                mask_disc = np.concatenate((masks_cat_batch, img_batch),
                                           axis=3)
                # train_masks_cat_new = np.concatenate((train_masks_cat, train_imgs), axis=3)
                mask_disc = patch_utils.extract_patches(
                    images=mask_disc, sub_patch_dim=sub_patch_dim)
                # patch_utils.get_disc_batch
                gen_mask_disc = generator_nn.predict(img_batch)
                gen_mask_disc = np.concatenate((gen_mask_disc, img_batch),
                                               axis=3)
                print("gen_mask_disc:", gen_mask_disc.shape)
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
                # counts batches we've ran through for generating fake vs real images
                batch_counter += 1

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

    # patch GAN model
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
                          patch_dim=sub_patch_dim)
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
            batch_counter = 1
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
                # counts batches we've ran through for generating fake vs real images
                batch_counter += 1

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

    # pretrained unet CNN model
    elif use_pretrained_unet_cnn:
        print("---------use_cnn---------")
        optimizer = RMSprop(lr=0.002, clipvalue=1.0, decay=1e-8)
        discriminator = build_discriminator(mask_shape)
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=opt_discriminator)
        discriminator.summary()
        generator_nn.compile(loss='categorical_crossentropy',
                             optimizer=opt_discriminator)
        img = Input(shape=img_shape)
        rec_mask = generator_nn(img)

        discriminator.trainable = False
        validity = discriminator(rec_mask)
        adversarial_model = Model(img, [rec_mask, validity], name='D')
        adversarial_model.compile(
            loss=['categorical_crossentropy', 'binary_crossentropy'],
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
        for epoch in range(0, total_epochs):
            generator_nn.fit(train_imgs,
                             train_masks_cat,
                             batch_size=batch_size,
                             epochs=1,
                             verbose=True,
                             shuffle=True)
            print('Epoch', str(epoch), '/', str(total_epochs))

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
                    '_pretrained_unet_cnn_pred_epoch_' + str(current_epoch) +
                    '.nii.gz')
                sitk.WriteImage(
                    sitk.GetImageFromArray(orig_mask_2), output + '/' +
                    test_imgs_np_file_2[0:6] + '_pretrained_unet_cnn_epoch_' +
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

    # CNN model
    elif use_cnn:
        print("---------use_cnn---------")
        optimizer = RMSprop(lr=0.002, clipvalue=1.0, decay=1e-8)
        discriminator = build_discriminator(mask_shape)
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=opt_discriminator)
        discriminator.summary()
        generator_nn.compile(loss='categorical_crossentropy',
                             optimizer=opt_discriminator)
        img = Input(shape=img_shape)
        rec_mask = generator_nn(img)

        discriminator.trainable = False
        validity = discriminator(rec_mask)
        adversarial_model = Model(img, [rec_mask, validity], name='D')
        adversarial_model.compile(
            loss=['categorical_crossentropy', 'binary_crossentropy'],
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
        optimizer = RMSprop(lr=0.002, clipvalue=1.0, decay=1e-8)

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

        generator_img.compile(loss='categorical_crossentropy',
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
