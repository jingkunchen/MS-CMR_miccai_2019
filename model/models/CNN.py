from keras.layers import Input, Dense, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU


def build_discriminator(input_shape):
        """Build the discriminator/D network
        
        Arguments:
            input_shape {list} -- Input tensor shape of the discriminator network, either the real unmodified image
                or the generated image by generator/R network.
        
        Returns:
            [Tensor] -- Network output tensors.
        """
        df_dim = 16
        image = Input(shape=input_shape, name='d_input')
        x = Conv2D(filters=df_dim, kernel_size = 5, strides=2, padding='same', name='d_h0_conv')(image)
        x = LeakyReLU()(x)

        x = Conv2D(filters=df_dim*2, kernel_size = 5, strides=2, padding='same', name='d_h1_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=df_dim*4, kernel_size = 5, strides=2, padding='same', name='d_h2_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=df_dim*8, kernel_size = 5, strides=2, padding='same', name='d_h3_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # x = Conv2D(filters=df_dim*16, kernel_size = 5, strides=2, padding='same', name='d_h4_conv')(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)

        # x = Conv2D(filters=df_dim*32, kernel_size = 5, strides=2, padding='same', name='d_h5_conv')(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)

        # x = Conv2D(filters=df_dim*64, kernel_size = 5, strides=2, padding='same', name='d_h6_conv')(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid', name='d_h3_lin')(x)

        return Model(image, x, name='D')