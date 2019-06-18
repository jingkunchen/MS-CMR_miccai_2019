from __future__ import division
from __future__ import print_function

import os
import numpy as np
import SimpleITK as sitk

from data_gen import prep_img
from models.DRUNet32f import get_model


def gen_masks(model, input_img_fold, flair = False, t1 = False, normilize=True, output_masks_fold=''):
    assert flair or t1

    assert os.path.isdir(input_img_fold)

    if output_masks_fold != '':
        assert os.path.isdir(output_masks_fold)
        if output_masks_fold[-1] != '/':
            output_masks_fold += '/'

    if output_masks_fold[-1] != '/':
        output_masks_fold += '/'

    if flair:
        flair_files = os.listdir(os.path.join(input_img_fold, 'FLAIR'))
    if t1:
        t1_img_files = os.listdir(os.path.join(input_img_fold, 'T1'))

    if flair:
        img_files = flair_files
    elif t1:
        img_files = t1_img_files

    masks = []
    for ind, img_file in enumerate(img_files):
        img = None
        if flair:
            flair_img = sitk.ReadImage(os.path.join(input_img_fold, 'FLAIR/' + img_file.split('_')[0]+'_FLAIR.nii'))
            flair_img = sitk.GetArrayFromImage(flair_img)
            flair_img = flair_img.astype('float32')
            flair_img = np.nan_to_num(flair_img)
            flair_img = prep_img(flair_img, normilize=normilize)
            flair_img = flair_img[..., np.newaxis]
            img = flair_img
        if t1:
            t1_img = sitk.ReadImage(os.path.join(input_img_fold, 'T1/' + img_file.split('_')[0]+'_T1.nii'))
            t1_img = sitk.GetArrayFromImage(t1_img)
            t1_img = t1_img.astype('float32')
            t1_img = np.nan_to_num(t1_img)
            t1_img = prep_img(t1_img, normilize=normilize)
            t1_img = t1_img[..., np.newaxis]
            img = t1_img

        if flair and t1:
            img = np.concatenate((flair_img, t1_img), axis=3)

        pred_mask = model.predict(img)
        pred_mask = pred_mask.argmax(axis=3)

        pred_mask = pred_mask.astype('float32')

        masks.append(pred_mask)

        if output_masks_fold != '':
            sitk.WriteImage(sitk.GetImageFromArray(pred_mask),
                            output_masks_fold + img_file.split('_')[0] + '_mask.nii')

    masks = np.array(masks, dtype='float32')
    return masks


def main():
    input_img_fold = 'data/Utrecht/images'
    output_mask_fold = 'data/Utrecht/masks'
    pretrained_model = 'pretrained_models/weights_dru32.h5'
    img_shape = (240, 240, 2)
    num_classes = 9
    model = get_model(img_shape=img_shape, num_classes=num_classes)
    model.load_weights(pretrained_model)
    masks = gen_masks(model, input_img_fold, flair=True, t1=True, normilize=True, output_masks_fold=output_mask_fold)

    return 0


if __name__ == '__main__':
    main()