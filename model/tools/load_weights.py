from __future__ import print_function
from __future__ import division


from models.DRUNet32f import get_model


def main():

    old_model = get_model(img_shape=(240, 240, 2), num_classes=1)
    old_model.load_weights('pretrained_models/0.h5')

    model = get_model(img_shape=(240, 240, 2), num_classes=9)

    for ind, layer in enumerate(old_model.layers[1:10]):
        if layer.trainable:
            model.layers[ind+1].set_weights(layer.get_weights())

    model.save_weights('pretrained_models/pretrained_drunet.h5')

    return 0

if __name__ == '__main__':
    main()