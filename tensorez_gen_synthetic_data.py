import numpy as np
import tensorflow as tf
import os
import glob
from tensorez.util import *
from tensorez.model import *
from tensorez.bayer import *

batch_size = 5
output_path = os.path.join('data', 'synthetic')
psf_size = 128

def load_psfs():
    filename = os.path.join('data', 'ASICAP', 'CapObj', '2020-07-30Z', '2020-07-30-2005_0-CapObj.SER')
    crop_center = (1287, 993)

    os.makedirs(output_path, exist_ok = True)

    frames = []
    for frame_index in range(0, batch_size):
        frame = read_image(filename, frame_index = frame_index, crop = (psf_size, psf_size), crop_center = crop_center)

        frame_max = tf.math.reduce_max(frame, axis = (-2, -3), keepdims = True)
        print(f"raw psf max is {frame_max}")

        frame_min = tf.math.reduce_min(frame, axis = (-2, -3), keepdims = True)
        frame = frame - frame_min
        frame_sum = tf.math.reduce_sum(frame, axis = (-2, -3), keepdims = True)
        frame = frame / frame_sum



        write_image(frame, os.path.join(output_path, f"psf{frame_index}.png"), normalize = True)


        frames.append(frame)

    frames = tf.concat(frames, axis = -4)
    print(f"frames.shape {frames.shape}")
    return frames



tf.compat.v1.enable_eager_execution()

psfs = load_psfs()

true_image_filename = os.path.join('misc_images', 'STS-115_ISS_after_undocking.jpg')
true_image = read_image(true_image_filename)
downsample = 2

true_image = tf.image.resize(true_image, (true_image.shape[-3] // downsample, true_image.shape[-2] // downsample), method = tf.image.ResizeMethod.BICUBIC)
#true_image = tf.squeeze(true_image, axis = -4)

model = TensoRezModel(psf_size = psfs.shape[-2], super_resolution_factor= 1, realign_center_of_mass = False, model_demosaic = False)
model.build((batch_size, true_image.shape[-3], true_image.shape[-2], true_image.shape[-1]))
model.estimated_image = tf.pad(true_image, ((0, 0), (psf_size // 2, psf_size // 2), (psf_size // 2, psf_size // 2), (0, 0)))
model.point_spread_functions = tf.reshape(psfs, model.point_spread_functions.shape)

predicted_images = model.predict_observed_images()

for frame_index in range(0, batch_size):
    write_image(predicted_images[frame_index,...], os.path.join(output_path, f'prediction{frame_index}.png'))



