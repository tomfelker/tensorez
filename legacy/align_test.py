import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

import os
import os.path
import tensorez.align as align
from tensorez.util import *
from tensorez.image_sequence import *
from tensorez.align import *
import gc
import glob


output_path = os.path.join('output', 'align_test')

lights = ImageSequence(os.path.join('data', '2022-01-11_jwst', '2022-01-11-0941_0-CapObj.SER'), frame_step = 1)
darks = ImageSequence(os.path.join('data', '2022-01-11_jwst', '2022-01-11-1027_1-CapObj.SER'), frame_step = 1)

#file_glob = os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'lights', '*.SER'); frame_step = 10
#file_glob_darks = None # os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'darks', '*.SER');


dark_image = None
if darks is not None:
    dark_image = darks.read_average_image()

alignment_transforms = compute_alignment_transforms(lights, dark_image=dark_image, debug_output_dir=output_path)

final_average_image_bhwc = read_average_image_with_alignment_transforms(lights, alignment_transforms, lights[0].shape, dark_image)
write_image(final_average_image_bhwc, os.path.join(output_path, 'final_average_image.png'))


for image_index, image_bhwc in enumerate(lights):
    if dark_image is not None:
        image_bhwc -= dark_image

    alignment_transform = alignment_transforms[image_index, :]
    transformed_image_bhwc = align.transform_image(image_bhwc, alignment_transform)
    write_image(transformed_image_bhwc, os.path.join(output_path, "final_aligned_{:08d}.png".format(image_index)))