
"""Architectures de modeles pour la segmentation semantique."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam


def conv_block(inputs, n_filters, kernel_size=3, batch_norm=True):
    """Bloc de convolution double."""
    x = layers.Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal'')(inputs)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu'')(x)
    x = layers.Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal'')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu'')(x)
    return x


def encoder_block(inputs, n_filters, dropout_rate=0.3):
    """Bloc encodeur."""
    skip = conv_block(inputs, n_filters)
    pool = layers.MaxPooling2D((2, 2))(skip)
    pool = layers.Dropout(dropout_rate)(pool)
    return skip, pool


def decoder_block(inputs, skip, n_filters, dropout_rate=0.3):
    """Bloc decodeur."""
    x = layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same'')(inputs)
    x = layers.Concatenate()([x, skip])
    x = layers.Dropout(dropout_rate)(x)
    x = conv_block(x, n_filters)
    return x


def build_unet(input_shape, n_classes, filters=[32, 64, 128, 256, 512]):
    """Construit un modele U-Net."""
    inputs = layers.Input(shape=input_shape)
    
    skip1, pool1 = encoder_block(inputs, filters[0])
    skip2, pool2 = encoder_block(pool1, filters[1])
    skip3, pool3 = encoder_block(pool2, filters[2])
    skip4, pool4 = encoder_block(pool3, filters[3])
    
    bottleneck = conv_block(pool4, filters[4])
    
    dec4 = decoder_block(bottleneck, skip4, filters[3])
    dec3 = decoder_block(dec4, skip3, filters[2])
    dec2 = decoder_block(dec3, skip2, filters[1])
    dec1 = decoder_block(dec2, skip1, filters[0])
    
    outputs = layers.Conv2D(n_classes, (1, 1), activation='softmax'')(dec1)
    
    return Model(inputs, outputs, name='UNet')


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Coefficient de Dice."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Dice loss."""
    return 1 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """CCE + Dice loss."""
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return cce + dice
