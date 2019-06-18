from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, Concatenate
from keras import backend as K
K.set_image_data_format('channels_last')

def get_model(img_shape=None, num_classes=2):
    inputs = Input(shape=img_shape)

    conv1_1 = Conv2D(32, (3, 3), dilation_rate=1, padding='same', activation='relu')(inputs)
    conv1_2 = Conv2D(32, (3, 3), dilation_rate=1, padding='same', activation='relu')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv2_1 = Conv2D(32, (3, 3), dilation_rate=2, padding='same', activation='relu')(pool1)
    conv2_skip = Conv2D(32, (1, 1), dilation_rate=2, padding='same', activation='relu')(pool1)
    conv2_2 = Conv2D(32, (3, 3), dilation_rate=2, padding='same', activation='relu')(conv2_1)
    add2 = Add()([conv2_2, conv2_skip])
    pool2 = MaxPooling2D(pool_size=(2, 2))(add2)

    conv3_1 = Conv2D(32, (3, 3), dilation_rate=3, padding='same', activation='relu')(pool2)
    conv3_skip = Conv2D(32, (1, 1), dilation_rate=3, padding='same', activation='relu')(pool2)
    conv3_2 = Conv2D(32, (3, 3), dilation_rate=3, padding='same', activation='relu')(conv3_1)
    add3 = Add()([conv3_2, conv3_skip])
    pool3 = MaxPooling2D(pool_size=(2, 2))(add3)

    conv4_1 = Conv2D(32, (3, 3), dilation_rate=4, padding='same', activation='relu')(pool3)
    conv4_skip = Conv2D(32, (1, 1), dilation_rate=4, padding='same', activation='relu')(pool3)
    conv4_2 = Conv2D(32, (3, 3), dilation_rate=4, padding='same', activation='relu')(conv4_1)
    add4 = Add()([conv4_2, conv4_skip])
    upsamp4 = UpSampling2D(size=(2, 2))(add4)

    skip_concat1 = Concatenate()([upsamp4, add3])

    conv5_1 = Conv2D(32, (3, 3), dilation_rate=3, padding='same', activation='relu')(skip_concat1)
    conv5_skip = Conv2D(32, (1,1), dilation_rate=3, padding='same', activation='relu')(skip_concat1)
    conv5_2 = Conv2D(32, (3, 3), dilation_rate=3, padding='same', activation='relu')(conv5_1)
    add5 = Add()([conv5_2, conv5_skip])
    upsamp5 = UpSampling2D(size=(2, 2))(add5)

    skip_concat2 = Concatenate()([upsamp5, add2])

    conv6_1 = Conv2D(32, (3, 3), dilation_rate=2, padding='same', activation='relu')(skip_concat2)
    conv6_skip = Conv2D(32, (1, 1), dilation_rate=2, padding='same', activation='relu')(skip_concat2)
    conv6_2 = Conv2D(32, (3, 3), dilation_rate=2, padding='same', activation='relu')(conv6_1)
    add6 = Add()([conv6_2, conv6_skip])
    upsamp6 = UpSampling2D(size=(2, 2))(add6)

    skip_concat3 = Concatenate()([upsamp6, conv1_2])

    conv7_1 = Conv2D(32, (3, 3), dilation_rate=1, padding='same', activation='relu')(skip_concat3)
    conv7_2 = Conv2D(32, (3, 3), dilation_rate=1, padding='same', activation='relu')(conv7_1)

    conv8 = Conv2D(num_classes, (1, 1), padding='same', activation='softmax')(conv7_2)

    model = Model(inputs=inputs, outputs=conv8)

    return model