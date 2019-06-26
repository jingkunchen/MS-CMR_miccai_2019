from keras.layers import Input, Dense, Flatten, BatchNormalization, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers


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
        x = Conv2D(filters=df_dim, kernel_size = 5, strides=1, padding='same', name='d_h0_conv', kernel_regularizer=regularizers.l2(0.01))(image)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=df_dim*2, kernel_size = 5, strides=1, padding='same', name='d_h1_conv', kernel_regularizer=regularizers.l2(0.01))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=df_dim*4, kernel_size = 5, strides=1, padding='same', name='d_h2_conv', kernel_regularizer=regularizers.l2(0.01))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=df_dim*8, kernel_size = 5, strides=1, padding='same', name='d_h3_conv', kernel_regularizer=regularizers.l2(0.01))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid', name='d_h3_lin')(x)

        return Model(image, x, name='D')