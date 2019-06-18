from __future__ import print_function
from __future__ import division

import numpy as np
# from keras.preprocessing.image import apply_transform, transform_matrix_offset_center


from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import transform_matrix_offset_center


def augmentation(x, y):
    theta = (np.random.uniform(-15, 15) * np.pi) / 180.
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    shear = np.random.uniform(-.1, .1)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    zx, zy = np.random.uniform(.95, 1.05, 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    augmentation_matrix = np.dot(np.dot(rotation_matrix, shear_matrix), zoom_matrix)
    img_gen = ImageDataGenerator()
    # transform_matrix = transform_matrix_offset_center(augmentation_matrix, x[0].shape[0], x[0].shape[1])
    transform_matrix = img_gen.transform_matrix_offset_center(augmentation_matrix, x[0].shape[0], x[0].shape[1])

    x_aug = np.zeros(x.shape, dtype=np.float32)
    
    for chan in range(x.shape[-1]):
        # x_aug[:, :, chan:chan+1] = apply_transform(x[:, :, chan, np.newaxis], transform_matrix, channel_axis=2)
        x_aug[:, :, chan:chan+1] = img_gen.apply_transform(x[:, :, chan, np.newaxis], transform_matrix, channel_axis=2)
        
    # x_0 = apply_transform(x_0[..., np.newaxis], transform_matrix, channel_axis=2)
    # x_1 = apply_transform(x_1[..., np.newaxis], transform_matrix, channel_axis=2)
    
    y_aug = img_gen.apply_transform(y, transform_matrix, channel_axis=2)
    # y_aug = apply_transform(y, transform_matrix, channel_axis=2)

    return x_aug, y_aug

def main():
    return 0

if __name__ == '__main__':
    main()