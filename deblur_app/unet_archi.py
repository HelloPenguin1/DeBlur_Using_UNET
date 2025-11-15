# deblur_app/unet_architecture.py

import tensorflow as tf
from keras import layers, models

def encoder_block(input_tensor, num_filters):
    """Creates an encoder block (Conv -> BN -> ReLU -> Conv -> ReLU)."""
    c = layers.Conv2D(num_filters, 3, activation='relu', padding='same')(input_tensor)
    c = layers.BatchNormalization()(c)
    c = layers.Conv2D(num_filters, 3, activation='relu', padding='same')(c)
    p = layers.MaxPooling2D((2, 2))(c)
    return c, p  # Return the block output (for skip connection) and the pooled output

def decoder_block(input_tensor, skip_tensor, num_filters):
    """Creates a decoder block (UpSample -> Concat -> Conv -> ReLU -> Conv -> ReLU)."""
    u = layers.UpSampling2D((2, 2))(input_tensor)
    u = layers.Concatenate()([u, skip_tensor])
    c = layers.Conv2D(num_filters, 3, activation='relu', padding='same')(u)
    c = layers.Conv2D(num_filters, 3, activation='relu', padding='same')(c)
    return c

def build_deblurring_cnn(input_shape=(256, 256, 3)):
    """
    Build your custom U-Net deblurring architecture.
    Note: Using 128x128 to match your config, adjust if needed.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1, p1 = encoder_block(inputs, 64)
    c2, p2 = encoder_block(p1, 128)

    # Bottleneck
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(b)

    # Decoder
    d1 = decoder_block(b, c2, 128)
    d2 = decoder_block(d1, c1, 64)

    outputs = layers.Conv2D(3, 1, activation='linear', padding='same')(d2)

    # Output + residual connection
    outputs = layers.Add()([outputs, inputs])
    outputs = layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0))(outputs)

    model = models.Model(inputs, outputs, name="U-Net_DeblurringCNN")
    
    return model
