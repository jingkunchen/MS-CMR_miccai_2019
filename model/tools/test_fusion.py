from __future__ import division
from __future__ import print_function

import numpy as np

from run_test import get_eval_metrics
from models.DRUNet32f import get_model


def main():
    pretrained_model = 'pretrained_models/weights_dru32_70.h5'
    test_imgs_np_file = 'data/np_data/test_case_70/test/images.npy'
    test_masks_np_file = 'data/np_data/test_case_70/test/masks.npy'

    img_shape = (240, 240, 2)
    num_classes = 9

    dru32_8c_model = get_model(img_shape=img_shape, num_classes=num_classes)
    dru32_8c_model.load_weights(pretrained_model)

    test_imgs = np.load(test_imgs_np_file)
    test_masks = np.load(test_masks_np_file)
    test_masks = test_masks[:, :, :, 0]

    pred_masks = dru32_8c_model.predict(test_imgs)
    pred_masks = pred_masks.argmax(axis=3)
    dsc, h95, vs = get_eval_metrics(test_masks, pred_masks)

    print("Evaluation on pure multiclass model. DRUNet32f\n")
    print('DSC:')
    print(dsc)
    print("\nH95:")
    print(h95)
    print("\nVS")
    print(vs)

    c4_pred_masks = np.load('evaluation/masks_label4/label_4_case_70.npy')
    c4_pred_masks = c4_pred_masks.transpose((0, 2, 1))
    fuse4_pred_masks = pred_masks.copy()
    fuse4_pred_masks[fuse4_pred_masks == 4] = 3
    fuse4_pred_masks[c4_pred_masks==1] = 4

    fuse4_dsc, fuse4_h95, fuse4_vs = get_eval_metrics(test_masks, fuse4_pred_masks)

    print("Evaluation after fusing 4th label. DRUNet32f\n")
    print('Fuse4 DSC:')
    print(fuse4_dsc)
    print("\nFuse4 H95:")
    print(fuse4_h95)
    print("\nFuse4 VS")
    print(fuse4_vs)


    c5_dru32_model = get_model(img_shape, 2)
    c5_dru32_model.load_weights('pretrained_models/weights_c5dru32_70.h5')
    c5_imgs = np.load('data/np_data/t1_ir_data/test_case_70/test/images.npy')
    c5_pred_masks = c5_dru32_model.predict(c5_imgs)
    c5_pred_masks = c5_pred_masks.argmax(axis=3)

    fuse5_pred_masks = pred_masks.copy()
    fuse5_pred_masks[c5_pred_masks==1] = 5

    fuse5_dsc, fuse5_h95, fuse5_vs = get_eval_metrics(test_masks, fuse5_pred_masks)

    print("Evaluation after fusing 5th label. DRUNet32f\n")
    print('Fuse5 DSC:')
    print(fuse5_dsc)
    print("\nFuse5 H95:")
    print(fuse5_h95)
    print("\nFuse5 VS")
    print(fuse5_vs)


    fuse4_fuse5_pred_masks = fuse4_pred_masks.copy()
    fuse4_fuse5_pred_masks[c5_pred_masks==1] = 5

    fuse4_fuse5_dsc, fuse4_fuse5_h95, fuse4_fuse5_vs = get_eval_metrics(test_masks, fuse4_fuse5_pred_masks)

    print("Evaluation after fusing 5th label. DRUNet32f\n")
    print('Fuse4 Fuse5 DSC:')
    print(fuse4_fuse5_dsc)
    print("\nFuse4 Fuse5 H95:")
    print(fuse4_fuse5_h95)
    print("\nFuse4 Fuse5 VS")
    print(fuse4_fuse5_vs)

    return 0


if __name__ == '__main__':
    main()