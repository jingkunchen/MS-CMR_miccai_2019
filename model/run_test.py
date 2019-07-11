from __future__ import print_function
from __future__ import division

import click
import json
import os
import numpy as np
import SimpleITK as sitk
from keras.optimizers import Adam
from evaluation import getDSC, getHausdorff, getVS
from models.DRUNet32f import get_model
from metrics import dice_coef, dice_coef_loss

# label transform, 500-->1, 200-->2, 600-->3

def get_eval_metrics(true_mask, pred_mask, output_file=''):
    true_mask_sitk = sitk.GetImageFromArray(true_mask)
    pred_mask_sitk = sitk.GetImageFromArray(pred_mask)
    dsc = getDSC(true_mask_sitk, pred_mask_sitk)
    h95 = getHausdorff(true_mask_sitk, pred_mask_sitk)
    vs = getVS(true_mask_sitk, pred_mask_sitk)

    result = {}
    result['dsc'] = dsc
    result['h95'] = h95
    result['vs'] = vs

    if output_file != '':
        with open(output_file, 'w+') as outfile:
            json.dump(result, outfile)

    return (dsc, h95, vs)


@click.command()
@click.argument('test_imgs_np_file', type=click.STRING)
@click.argument('test_masks_np_file', type=click.STRING)
@click.argument('pretrained_model', type=click.STRING)
@click.option('--output_pred_mask_file', type=click.STRING, default='')
@click.option('--output_metric_file', type=click.STRING, default='')
def main(test_imgs_np_file, test_masks_np_file, pretrained_model, output_pred_mask_file='', output_metric_file=''):
    num_classes = 9
    # learn_rate = 1e-5

    test_imgs = np.load(test_imgs_np_file)

    test_masks = np.load(test_masks_np_file)
    test_masks = test_masks[:, :, :, 0]

    img_shape = (test_imgs.shape[1], test_imgs.shape[2], 1)
    model = get_model(img_shape=img_shape, num_classes=num_classes)
    assert os.path.isfile(pretrained_model)
    model.load_weights(pretrained_model)
    pred_masks = model.predict(test_imgs)
    pred_masks = pred_masks.argmax(axis=3)
    dsc, h95, vs = get_eval_metrics(test_masks, pred_masks, output_metric_file)

    if output_pred_mask_file != '':
        np.save(output_pred_mask_file, pred_masks)

    return (dsc, h95, vs)

if __name__ == '__main__':
    main()
