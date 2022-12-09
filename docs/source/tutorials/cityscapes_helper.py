from PIL import Image
import numpy as np
import h5py
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, concatenate, MaxPool2D, Input, Dropout, UpSampling2D

class cityscapes:
    array_size = 256, 128

    def image_to_rgb_array(filename):
        img = Image.open(filename).resize(cityscapes.array_size, resample=Image.Resampling.NEAREST)
        return np.asarray(img.convert('RGB'))

    def append_to_array(arr, data):
        if arr is None:
            return np.array([data])
        else:
            return np.vstack([arr, [data]])

    def find_seg(rgb):
        global seg_map
        rows, _ = seg_map.shape
        for i in range(rows):
            if (seg_map[i] == rgb).all():
                return i
        seg_map = cityscapes.append_to_array(seg_map, rgb)
        return rows

    def seg_array(seg_rgb):
        seg_arr = np.array([np.array([np.array([cityscapes.find_seg(col)]) for col in row]) for row in seg_rgb])
        return seg_arr

    def dataloader(data_path, city, count):
        print("Loading {count} images from {city}".format(count=count, city=city))
        all_rgb = None
        all_seg = None
        global seg_map
        seg_map = np.array([[0, 0, 0]])
        for i in range(count):
            rgb_file = os.path.join(data_path, "leftImg8bit/train/{city}/{city}_{number}_000019_leftImg8bit.png".format(city=city, number=str(i).zfill(6)))
            seg_file = os.path.join(data_path, "gtFine/train/{city}/{city}_{number}_000019_gtFine_color.png".format(city=city, number=str(i).zfill(6)))

            rgb_arr = cityscapes.image_to_rgb_array(rgb_file)
            seg_rgb_arr = cityscapes.image_to_rgb_array(seg_file)
            seg_arr = cityscapes.seg_array(seg_rgb_arr)
            all_rgb = cityscapes.append_to_array(all_rgb, rgb_arr)
            all_seg = cityscapes.append_to_array(all_seg, seg_arr)
            print("Loaded image #{number} -- Total segments: {segments}".format(number=i, segments=seg_map.shape[0]))
        return (all_rgb, all_seg, seg_map)
    
    def load_data(data_path, city, count):
        all_rgb, all_seg, seg_map = cityscapes.dataloader(data_path, city, count)
        return (all_rgb/255., all_seg, seg_map)

def residual_segnet(input_size, drop_rate=0.05, n_out=1):
    inputs = Input(input_size)

    # Encoder network
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = Dropout(drop_rate)(conv1)

    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = Dropout(drop_rate)(conv2)

    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = Dropout(drop_rate)(conv3)

    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    conv4 = Dropout(drop_rate)(conv4)

    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    conv5 = Dropout(drop_rate)(conv5)

    # Decoder, upsample with skip connections
    up6 = Conv2D(512, 2, activation='relu',padding='same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(256, 2, activation='relu',padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(128, 2, activation='relu',padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(64, 2, activation='relu',padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    # conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)
    conv10 = Conv2D(n_out, 1, activation='softmax')(conv9)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv10)
    return model
