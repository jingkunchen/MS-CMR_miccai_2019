from __future__ import division

import os
import click
import numpy as np
import nibabel as nib

def prep_img(img, crop_shape=None, normilize=False):
    if crop_shape != None:
        assert len(crop_shape) == 2
        img_width = img.shape[0]
        img_height = img.shape[1]
        width = (img_width - crop_shape[0]) // 2
        height = (img_height - crop_shape[1]) // 2
        img = img[width:img_width - width, height:img_height - height]
    if normilize:
        mean = np.mean(img)
        std = np.std(img)
        img -= mean
        img /= std
    return img

def get_2d_patches(input_fold, test_img='', drop_class = [], crop_shape=None,
                   flair=False, t1=True, ir=True, normilize_per_case=True, output_fold=''):
    """
    Generating 2d patches based on 3d input images.
    Args:
        input_fold: string. Path to the folder with two other folders: images and masks.
                            In turn raw 3d images and masks are stored respectively.
        drop_class: list of integers. List of classes that will be reasigned to background (class 0).
        flair: boolean. True when flair images are used, False in opposite case.
        t1: boolean. True when t1 images are used, False in opposite case.
        ir: boolean. Ture when t1 images are used, Flase in opposite case.
        normilize_per_case: boolean. True when each image per case (3d image) is normilized, False if not.
        test_imgs:  string. Name of the image which is selected for testing among all images.
        output_fold: string. Path to the output folder with numpy files of images and masks.
                             Inside output folder train and test folder will be generated storing their images and masks respectively.
    Returns:
        tuple with 4 numpy files. 1.train images, 2.train masks, 3.test images, 4.test masks.
    """
    assert flair or t1 or ir

    assert os.path.isdir(input_fold)
    if output_fold != '':
        assert os.path.isdir(output_fold)
        if output_fold[-1] != '/':
            output_fold += '/'

    if input_fold[-1] != '/':
        input_fold += '/'

    if flair:
        flair_files = os.listdir(os.path.join(input_fold+"images/", 'FLAIR'))
    if t1:
        t1_img_files = os.listdir(os.path.join(input_fold + "images/", 'T1'))
    if ir:
        ir_img_files = os.listdir(os.path.join(input_fold + "images/", 'IR'))

    # if flair and t1:
    #     assert len(flair_files) == len(t1_img_files)
    #     img_files = flair_files
    if flair:
        img_files = flair_files
    elif t1:
        img_files = t1_img_files
    elif ir:
        img_files = ir_img_files

    train_imgs = []
    train_masks = []
    test_imgs = []
    test_masks = []

    for ind, img_file in enumerate(img_files):
        is_train_img = False
        if img_file.split('_')[0] != test_img:
            is_train_img = True

        img = None
        if flair:
            flair_img = nib.load(os.path.join(input_fold, 'images/FLAIR/' + img_file.split('_')[0]+'_FLAIR.nii')).get_data()
            flair_img = np.nan_to_num(flair_img)
            flair_img = flair_img.astype('float32')
            flair_img = prep_img(flair_img, crop_shape=crop_shape, normilize=normilize_per_case)
            flair_img = flair_img[..., np.newaxis]
            img = flair_img
        if t1:
            t1_img = nib.load(os.path.join(input_fold, 'images/T1/' + img_file.split('_')[0]+'_T1.nii')).get_data()
            t1_img = np.nan_to_num(t1_img)
            t1_img = t1_img.astype('float32')
            t1_img = prep_img(t1_img, crop_shape=crop_shape, normilize=normilize_per_case)
            t1_img = t1_img[..., np.newaxis]
            img = t1_img
        if ir:
            ir_img = nib.load(os.path.join(input_fold, 'images/IR/' + img_file.split('_')[0] + '_IR.nii')).get_data()
            ir_img = np.nan_to_num(ir_img)
            ir_img = ir_img.astype('float32')
            ir_img = prep_img(ir_img, crop_shape=crop_shape, normilize=normilize_per_case)
            ir_img = ir_img[..., np.newaxis]
            img = ir_img

        if flair and t1 and ir:
            img = np.concatenate((flair_img, t1_img, ir_img), axis=3)
        elif flair and t1:
            img = np.concatenate((flair_img, t1_img), axis=3)
        elif flair and ir:
            img = np.concatenate((flair_img, ir_img), axis=3)
        elif t1 and ir:
            img = np.concatenate((t1_img, ir_img), axis=3)

        mask = nib.load(os.path.join(input_fold, 'masks/' + img_file.split('_')[0] + '_mask.nii')).get_data()
        mask = np.nan_to_num(mask)
        mask = prep_img(mask, crop_shape=crop_shape, normilize=False)

        for elem in drop_class:
            mask[mask == elem] = 0
        # mask[mask==5] = 1

        for layer in range(img.shape[2]):
            patch_img = img[:, :, layer]
            patch_mask = mask[:, :, layer]
            # patch_img = np.reshape(patch_img, (patch_img.shape[0], patch_img.shape[1], 1))
            patch_mask = np.reshape(patch_mask, (patch_mask.shape[0], patch_mask.shape[1], 1))
            if is_train_img:
                train_imgs.append(patch_img)
                train_masks.append(patch_mask)
            else:
                test_imgs.append(patch_img)
                test_masks.append(patch_mask)

    train_imgs = np.array(train_imgs, dtype='float32')
    train_masks = np.array(train_masks, dtype='float32')
    test_imgs = np.array(test_imgs, dtype='float32')
    test_masks = np.array(test_masks, dtype='float32')

    if output_fold != '':
        if os.path.isdir(os.path.join(output_fold, 'train')) == False:
            os.mkdir(os.path.join(output_fold, 'train'))
        if os.path.isdir(os.path.join(output_fold, 'test')) == False:
            os.mkdir(os.path.join(output_fold, 'test'))
        np.save(os.path.join(output_fold + 'train/', 'images'), train_imgs)
        np.save(os.path.join(output_fold + 'train/', 'masks'), train_masks)
        if test_img != '':
            np.save(os.path.join(output_fold + 'test/', 'images'), test_imgs)
            np.save(os.path.join(output_fold + 'test/', 'masks'), test_masks)

    return (train_imgs, train_masks, test_imgs, test_masks)

@click.command()
@click.argument('input_fold', type=click.STRING)
@click.option('--test_img', default='', type=click.STRING, help='Name of test image among all images')
@click.option('--flair', default=True, type=click.BOOL)
@click.option('--t1', default=True, type=click.BOOL)
@click.option('--ir', default=True, type=click.BOOL)
@click.option('--normilize_per_case', default=True, type=click.BOOL )
@click.option('--output_fold', default='', type=click.STRING, help='Path to the output folder where numpy arrays will be stored')
def main(input_fold, test_img, normilize_per_case, flair, t1, ir, output_fold):
    get_2d_patches(input_fold=input_fold, normilize_per_case=normilize_per_case, drop_class=[9, 10],
                   crop_shape=None, flair=flair, t1=t1, ir=ir, test_img=test_img, output_fold=output_fold)

if __name__ == '__main__':
    main()