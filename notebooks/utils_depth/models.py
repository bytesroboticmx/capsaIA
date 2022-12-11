# https://github.com/aamini/evidential-deep-learning/blob/main/neurips2020/models/depth/deterministic.py

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Cropping2D,
    concatenate,
    ZeroPadding2D,
    SpatialDropout2D,
)
import functools
from keras.utils.layer_utils import count_params

import utils_depth.config as config

########################## U-net ##########################


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = target.get_shape()[2] - refer.get_shape()[2]
    assert cw >= 0
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = target.get_shape()[1] - refer.get_shape()[1]
    assert ch >= 0
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)


# https://github.com/aamini/evidential-deep-learning/blob/main/neurips2020/models/depth/dropout.py
def unet(
    input_shape=(128, 160, 3),
    drop_prob=0.0,
    reg=None,
    activation=tf.nn.relu,
    num_class=1,
    compile=False,
):

    concat_axis = 3
    inputs = tf.keras.layers.Input(shape=input_shape)
    # inputs_normalized = tf.multiply(inputs, 1/255.)

    Conv2D_ = functools.partial(
        Conv2D,
        activation=activation,
        padding="same",
        kernel_regularizer=reg,
        bias_regularizer=reg,
    )

    conv1 = Conv2D_(32, (3, 3))(inputs)
    conv1 = Conv2D_(32, (3, 3))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = SpatialDropout2D(drop_prob)(pool1)

    conv2 = Conv2D_(64, (3, 3))(pool1)
    conv2 = Conv2D_(64, (3, 3))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = SpatialDropout2D(drop_prob)(pool2)

    conv3 = Conv2D_(128, (3, 3))(pool2)
    conv3 = Conv2D_(128, (3, 3))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = SpatialDropout2D(drop_prob)(pool3)

    conv4 = Conv2D_(256, (3, 3))(pool3)
    conv4 = Conv2D_(256, (3, 3))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = SpatialDropout2D(drop_prob)(pool4)

    conv5 = Conv2D_(512, (3, 3))(pool4)
    conv5 = Conv2D_(512, (3, 3))(conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch, cw))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = Conv2D_(256, (3, 3))(up6)
    conv6 = Conv2D_(256, (3, 3))(conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = Conv2D_(128, (3, 3))(up7)
    conv7 = Conv2D_(128, (3, 3))(conv7)

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D_(64, (3, 3))(up8)
    conv8 = Conv2D_(64, (3, 3))(conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D_(32, (3, 3))(up9)
    conv9 = Conv2D_(32, (3, 3))(conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
    conv10 = Conv2D(num_class, (1, 1))(conv9)

    # conv10 = tf.multiply(conv10, 255.)
    model = tf.keras.models.Model(inputs=inputs, outputs=conv10)

    if not compile:
        return model
    # for demo
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LR),
            loss="mse",
        )
        return model


########################## U-net separated on encoder and decoder (for vae) ##########################

Conv2D_ = functools.partial(
    Conv2D,
    activation=tf.nn.relu,
    padding="same",
    kernel_regularizer=None,
    bias_regularizer=None,
)


def print_num_params(model, name):
    trainable_count = count_params(model.trainable_weights)
    non_trainable_count = count_params(model.non_trainable_weights)
    print(f"{name}_trainable_count", trainable_count)
    print(f"{name}_non_trainable_count", non_trainable_count)


def get_encoder(input_shape=(128, 160, 3), out_units=100):

    inputs = tf.keras.layers.Input(shape=input_shape)  # (B, 128, 160, 3)
    print("encoder-inputs: ", inputs.shape)

    conv1 = Conv2D_(32, (3, 3))(inputs)
    conv1 = Conv2D_(32, (3, 3))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # (B, 64, 80, 32)
    print("encoder-pool1: ", pool1.shape)

    conv2 = Conv2D_(64, (3, 3))(pool1)
    conv2 = Conv2D_(64, (3, 3))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # (B, 32, 40, 64)
    print("encoder-pool2: ", pool2.shape)

    conv3 = Conv2D_(128, (3, 3))(pool2)
    conv3 = Conv2D_(128, (3, 3))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # (B, 16, 20, 128)
    print("encoder-pool3: ", pool3.shape)

    conv4 = Conv2D_(256, (3, 3))(pool3)
    conv4 = Conv2D_(256, (3, 3))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # (B, 8, 10, 256)
    print("encoder-pool4: ", pool4.shape)

    conv5 = Conv2D_(256, (3, 3))(pool4)
    conv5 = Conv2D_(256, (3, 3))(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)  # (B, 4, 5, 256)
    print("encoder-pool5: ", pool5.shape)

    conv6 = Conv2D_(256, (3, 3))(pool5)
    conv6 = Conv2D_(256, (3, 3))(conv6)  # (B, 4, 5, 256)
    print("encoder-conv6: ", conv6.shape)

    # bottleneck
    conv7 = Conv2D_(128, (3, 3))(conv6)
    # conv7 = Conv2D_(128, (3, 3))(conv7) # (B, 4, 5, 128)
    print("encoder-conv7: ", conv7.shape)

    conv8 = Conv2D_(64, (3, 3))(conv7)
    # conv8 = Conv2D_(64, (3, 3))(conv8) # (B, 4, 5, 64)
    print("encoder-conv8: ", conv8.shape)

    x = tf.keras.layers.Flatten()(conv8)  # (B, 4 * 5 * 64) -> (B, 1280)
    x = tf.keras.layers.Dense(256, activation="relu")(x)  # (B, 256)

    mean_layer = tf.keras.layers.Dense(out_units, activation=None)(x)  # (B, units)
    sigma_layer = tf.keras.layers.Dense(out_units, activation=tf.nn.softplus)(
        x
    )  # (B, units)

    encoder = tf.keras.models.Model(inputs, [mean_layer, sigma_layer])
    print_num_params(encoder, "encoder")
    return encoder


def get_decoder(input_shape=100, num_class=3):

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(160, activation="relu")(inputs)  # (B, 100) -> (B, 160)
    x = tf.keras.layers.Reshape((4, 5, -1))(x)  # (B, 4, 5, 8)

    conv8 = Conv2D_(64, (3, 3))(x)
    conv8 = Conv2D_(64, (3, 3))(conv8)  # (B, 4, 5, 64)
    print("decoder-conv8: ", conv8.shape)

    conv7 = Conv2D_(128, (3, 3))(conv8)
    conv7 = Conv2D_(128, (3, 3))(conv7)  # (B, 4, 5, 128)
    print("decoder-conv7: ", conv7.shape)

    conv6 = Conv2D_(256, (3, 3))(conv7)
    conv6 = Conv2D_(256, (3, 3))(conv6)  # (B, 4, 5, 256)
    print("decoder-conv6: ", conv6.shape)

    conv5 = Conv2D_(256, (3, 3))(conv6)
    # conv5 = Conv2D_(256, (3, 3))(conv5)
    print("decoder-conv5: ", conv5.shape)  # (B, 4, 5, 256)

    up_conv4 = UpSampling2D(size=(2, 2))(conv5)
    conv4 = Conv2D_(256, (3, 3))(up_conv4)
    conv4 = Conv2D_(256, (3, 3))(conv4)  # (B, 8, 10, 256)
    print("decoder-conv4: ", conv4.shape)

    up_conv3 = UpSampling2D(size=(2, 2))(conv4)
    conv3 = Conv2D_(128, (3, 3))(up_conv3)
    conv3 = Conv2D_(128, (3, 3))(conv3)  # (B, 16, 20, 128)
    print("decoder-conv3: ", conv3.shape)

    up_conv2 = UpSampling2D(size=(2, 2))(conv3)
    conv2 = Conv2D_(64, (3, 3))(up_conv2)
    conv2 = Conv2D_(64, (3, 3))(conv2)  # (B, 32, 40, 64)
    print("decoder-conv2: ", conv2.shape)

    up_conv1 = UpSampling2D(size=(2, 2))(conv2)
    conv1 = Conv2D_(32, (3, 3))(up_conv1)
    conv1 = Conv2D_(32, (3, 3))(conv1)  # (B, 64, 80, 32)
    print("decoder-conv1: ", conv1.shape)

    up_conv0 = UpSampling2D(size=(2, 2))(conv1)
    conv0 = Conv2D_(16, (3, 3))(up_conv0)
    conv0 = Conv2D_(16, (3, 3))(conv0)  # (B, 128, 160, 16)
    print("decoder-conv0: ", conv0.shape)

    out = Conv2D(num_class, (1, 1), activation=None)(conv0)  # sigmoid

    decoder = tf.keras.models.Model(inputs, out)
    print_num_params(decoder, "decoder")
    return decoder


if __name__ == "__main__":
    import numpy as np

    base_model = unet((128, 160, 3))
    x = np.ones((1, 128, 160, 3), dtype=np.float32)
    output = base_model(x)
    print(output.shape)
    print("Num params:", base_model.count_params())
