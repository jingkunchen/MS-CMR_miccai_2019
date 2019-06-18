from __future__ import print_function
from __future__ import division

import click
import json
import os
import numpy as np
from sklearn.utils import class_weight
from keras.optimizers import Adam
from keras.utils import to_categorical

from models.DRUNet32f import get_model
from run_test import get_eval_metrics
from tools.augmentation import augmentation
from metrics import weighted_categorical_crossentropy
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


@click.command()
@click.argument('train_imgs_np_file', type=click.STRING)
@click.argument('train_masks_np_file', type=click.STRING)
@click.argument('output_weights_file', type=click.STRING)
@click.option('--pretrained_model',
              type=click.STRING,
              default='',
              help='path to the pretrained model')
@click.option('--use_augmentation',
              type=click.BOOL,
              default=False,
              help='use data augmentation or not')
@click.option('--use_weighted_crossentropy',
              type=click.BOOL,
              default=False,
              help='use weighting of classes according to inbalance or not')
@click.option('--test_imgs_np_file',
              type=click.STRING,
              default='',
              help='path to the numpy file of test image')
@click.option('--test_masks_np_file',
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
         output_weights_file,
         pretrained_model='',
         use_augmentation=False,
         use_weighted_crossentropy=False,
         test_imgs_np_file='',
         test_masks_np_file='',
         output_test_eval=''):
    assert (test_imgs_np_file != '' and test_masks_np_file != '') or \
           (test_imgs_np_file == '' and test_masks_np_file == ''), \
        'Both test image file and test mask file must be given'

    num_classes = 4
    if not use_augmentation:
        # total_epochs = 1000
        total_epochs = 1
    else:
        total_epochs = 500
    batch_size = 16
    learn_rate = 2e-4

    eval_per_epoch = (test_imgs_np_file != '' and test_masks_np_file != '')
    if eval_per_epoch:
        test_imgs = np.load(test_imgs_np_file)
        test_masks = np.load(test_masks_np_file)

    train_imgs = np.load(train_imgs_np_file)
    print("train_imgs:",train_imgs.shape)
    train_masks = np.load(train_masks_np_file)
    print("train_masks:",train_masks.shape)

    if use_weighted_crossentropy:
        class_weights = class_weight.compute_class_weight(
            'balanced', np.unique(train_masks), train_masks.flatten())

    channels_num = train_imgs.shape[-1]
    img_shape = (train_imgs.shape[1], train_imgs.shape[2], channels_num)
    print("img_shape:",img_shape)
    model = get_model(img_shape=img_shape, num_classes=num_classes)
    if pretrained_model != '':
        assert os.path.isfile(pretrained_model)
        model.load_weights(pretrained_model)

    if use_augmentation:
        samples_num = train_imgs.shape[0]
        images_aug = np.zeros(train_imgs.shape, dtype=np.float32)
        masks_aug = np.zeros(train_masks.shape, dtype=np.float32)
        for i in range(samples_num):
            images_aug[i], masks_aug[i] = augmentation(train_imgs[i],
                                                       train_masks[i])

        train_imgs = np.concatenate((train_imgs, images_aug), axis=0)
        train_masks = np.concatenate((train_masks, masks_aug), axis=0)
    print("train_masks,num_classes:",train_masks.shape, num_classes)
    train_masks_cat = to_categorical(train_masks, num_classes)
    print("train_masks_cat:",train_masks_cat.shape)


    if use_weighted_crossentropy:
        model.compile(optimizer=Adam(lr=(learn_rate)),
                      loss=weighted_categorical_crossentropy(class_weights))
    else:
        model.compile(optimizer=Adam(lr=(learn_rate)),
                      loss='categorical_crossentropy')

    current_epoch = 1
    history = {}
    history['dsc'] = []
    history['h95'] = []
    history['vs'] = []
    while current_epoch <= total_epochs:
        print('Epoch', str(current_epoch), '/', str(total_epochs))
        model.fit(train_imgs,
                  train_masks_cat,
                  batch_size=batch_size,
                  epochs=1,
                  verbose=True,
                  shuffle=True)
                  
        if eval_per_epoch and current_epoch % 10 == 0:
            model.save_weights(output_weights_file)
            pred_masks = model.predict(test_imgs)
            pred_masks = pred_masks.argmax(axis=3)
            dsc, h95, vs = get_eval_metrics(test_masks[:, :, :, 0], pred_masks)
            history['dsc'].append(dsc)
            history['h95'].append(h95)
            history['vs'].append(vs)
            print(dsc)
            print(h95)
            print(vs)
            if output_test_eval != '':
                with open(output_test_eval, 'w+') as outfile:
                    json.dump(history, outfile)

        current_epoch += 1

    model.save_weights(output_weights_file)
    pred_masks = model.predict(train_imgs)
    pred_masks = pred_masks.argmax(axis=3)
    dsc, h95, vs = get_eval_metrics(train_masks[:, :, :, 0], pred_masks)
    print(dsc)
    print(h95)
    print(vs)

    if output_test_eval != '':
        with open(output_test_eval, 'w+') as outfile:
            json.dump(history, outfile)


if __name__ == "__main__":
    main()
