from keras import backend as K
from keras.layers import Input, Dense,  Flatten, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Cropping2D, ZeroPadding2D, Activation
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import os
import logging
from utils import *

smooth=1.
Input_height = 256
Input_width = 256
C_dim = 1 
Df_dim = 16
Filter = 5
Log_dir='log'
Sample_dir='sample'
Checkpoint_dir='checkpoint'
Epochs=5
Batch_size=16
Sample_interval=50

def model_dir():
        return "{}_{}".format(
            Input_height, Input_width)

def save(step, adversarial_model):
        """
        Helper method to save model weights.
        
        Arguments:
            step {[type]} -- [description]
        """
        os.makedirs(Checkpoint_dir, exist_ok=True)
        model_name = 'NVGAN_Model_{}.h5'.format(step)
        adversarial_model.save_weights(os.path.join(Checkpoint_dir, model_name))

def log_maker():
    log_dir = os.path.join(Log_dir, model_dir)
    os.makedirs(log_dir, exist_ok=True)

def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef_for_training(y_true, y_pred)

def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)
        return (ch1, ch2), (cw1, cw2)

#define U-Net architecture
def conv_bn_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs) #, kernel_initializer='he_normal'
    #bn = BatchNormalization()(conv)
    relu = Activation('relu')(conv)
    return relu

def build_generator(input_shape):
    """Build the generator/G network
    
    Arguments:
        input_shape {list} -- Input tensor shape of the generator network, either the real unmodified image.
    
    Returns:
        [Tensor] -- Network output tensors.
    """
    # u_net generator
    image = Input(shape = input_shape)
    concat_axis = -1
    # down sampling
    # 1
    conv1 = conv_bn_relu(64, Filter, image)
    conv1 = conv_bn_relu(64, Filter, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 2
    conv2 = conv_bn_relu(96, 3, pool1)
    conv2 = conv_bn_relu(96, 3, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 3
    conv3 = conv_bn_relu(128, 3, pool2)
    conv3 = conv_bn_relu(128, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 4
    conv4 = conv_bn_relu(256, 3, pool3)
    conv4 = conv_bn_relu(256, 4, conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # 5
    conv5 = conv_bn_relu(512, 3, pool4)
    conv5 = conv_bn_relu(512, 3, conv5)

    # up sampling
    # 5
    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    # 4
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = conv_bn_relu(256, 3, up6)
    conv6 = conv_bn_relu(256, 3, conv6)
    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    # 3
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = conv_bn_relu(128, 3, up7)
    conv7 = conv_bn_relu(128, 3, conv7)
    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    # 2
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = conv_bn_relu(96, 3, up8)
    conv8 = conv_bn_relu(96, 3, conv8)
    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    # 1
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = conv_bn_relu(64, 3, up9)
    conv9 = conv_bn_relu(64, 3, conv9)
    ch, cw = get_crop_shape(image, conv9)
    conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
    #  sigmoid to [-1,1]
    conv_end = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9) #, kernel_initializer='he_normal'
    return Model(image, outputs=conv_end, name = 'G')
        
def build_discriminator(input_shape):
    """Build the discriminator/D network
    
    Arguments:
        input_shape {list} -- Input tensor shape of the discriminator network, either the real unmodified image
            or the generated image by generator/R network.
    
    Returns:
        [Tensor] -- Network output tensors.
    """

    image = Input(shape=input_shape, name='d_input')
    x = Conv2D(filters=Df_dim, kernel_size =  5, strides=2, padding='same', name='d_h0_conv')(image)
    x = LeakyReLU()(x)

    x = Conv2D(filters=Df_dim*2, kernel_size = 5, strides=2, padding='same', name='d_h1_conv')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters=Df_dim*4, kernel_size = 5, strides=2, padding='same', name='d_h2_conv')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters=Df_dim*8, kernel_size = 5, strides=2, padding='same', name='d_h3_conv')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid', name='d_h3_lin')(x)
    return Model(image, x, name='D')

def train(images_label, images_unlabel, generator, discriminator, adversarial_model):
    log_maker()
    counter = 1
    plot_epochs = []
    plot_g_recon_losses = []
    # Adversarial ground truths
    ones = np.ones((Batch_size, 1))
    zeros = np.zeros((len(images_unlabel), 1))
    for epoch in range(Epochs):
        batch_idxs = len(images_label) // Batch_size
        for idx in range(0, batch_idxs):
            print('Epoch ({}/{})-------------------------------------------------'.format(epoch,Epochs)) 
            # Turn batch images data to float32 type.
            batch_images_label = np.array(images_label[idx * Batch_size:(idx + 1) * Batch_size]).astype(np.float32)
            images_unlabel = np.array(images_unlabel).astype(np.float32)

            generatored_batch_images_label = generator.predict(batch_images_label)
            generatored_images_unlabel = generator.predict(images_unlabel)
            # Update D network, minimize label images inputs->->R->D-> ones,  unlabel images->R->D->zeros loss.
            d_loss_real = discriminator.train_on_batch(generatored_batch_images_label, ones)
            d_loss_fake = discriminator.train_on_batch(generatored_images_unlabel, zeros)

            # Update R network, unlabel images->R->D->ones and reconstruction loss.
            g_loss = adversarial_model.train_on_batch(generatored_images_unlabel, [generatored_batch_images_label, ones])
            plot_epochs.append(epoch+idx/batch_idxs)
            plot_g_recon_losses.append(g_loss[1])
            counter += 1
            msg = 'Epoch:[{0}]-[{1}/{2}] --> d_loss: {3:>0.3f}, g_loss:{4:>0.3f}, g_recon_loss:{4:>0.3f}'.format(epoch, idx, batch_idxs, d_loss_real+d_loss_fake, g_loss[0], g_loss[1])
            print(msg)
            logging.info(msg)
            if np.mod(counter, Sample_interval) == 0:
                generatored_batch_images_label = generator.predict(batch_images_label)
                manifold_h = int(np.ceil(np.sqrt(generatored_batch_images_label.shape[0])))
                manifold_w = int(np.floor(np.sqrt(generatored_batch_images_label.shape[0])))
                save_images(generatored_batch_images_label, [manifold_h, manifold_w],
                    './{}/train_label_{:02d}_{:04d}.png'.format(Sample_dir, epoch, idx))
                generatored_images_unlabel = generator.predict(images_unlabel)
                manifold_h = int(np.ceil(np.sqrt(generatored_batch_images_label.shape[0])))
                manifold_w = int(np.floor(np.sqrt(generatored_batch_images_label.shape[0])))
                save_images(generatored_images_unlabel, [manifold_h, manifold_w],
                    './{}/train_unlabel_{:02d}_{:04d}.png'.format(Sample_dir, epoch, idx))
        save(epoch, adversarial_model)

def main():
    # load image
    images_label = np.load('images_label.npy')
    images_unlabel = np.load('images_unlabel.npy')

    # build model
    image_dims = [Input_height, Input_width, C_dim]
    img_tensor = Input(shape=image_dims)
    generator = build_generator(image_dims)
    generator.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)
    reconstructed_img = generator(img_tensor)

    optimizer = RMSprop(lr=0.002, clipvalue=1.0, decay=1e-8)
    discriminator = build_discriminator(image_dims)
    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')

    validity = discriminator(img_tensor)
    adversarial_model = Model(img_tensor, [reconstructed_img, validity])
    adversarial_model.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
            loss_weights=[0.2, 1],
            optimizer=optimizer)
    generator.summary()
    discriminator.summary()
    adversarial_model.summary()

    # train
    train(images_label, images_unlabel, generator, discriminator, adversarial_model)

if __name__=='__main__':
    main()