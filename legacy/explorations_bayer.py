# must be first! disable GPU so this stuff can be run without risk of interfering with long GPU runs
#except! you can't use CPU anyway, because apparently convolving by shape [3, 3, 1, 3] is a "grouped convolution" and isn't supported...
disable_gpu = False
if disable_gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import os
import glob
from tensorez.util import *
from tensorez.model import *
from tensorez.bayer import *

output_dir = os.path.join('output', 'experiments')

tf.enable_eager_execution()




example_image = read_image(os.path.join('misc_images', 'saturn20131017.jpg'))

shrink_by = 8
size_multiple_of = 2 * 3
example_image = tf.image.resizeaverage_image = tf.image.resize(example_image, (example_image.shape[-3] // shrink_by // size_multiple_of * size_multiple_of, example_image.shape[-2] // shrink_by // size_multiple_of * size_multiple_of), method = tf.image.ResizeMethod.BICUBIC)

write_image(example_image, os.path.join(output_dir, 'example_image.png'))



bayer_filtered_image = apply_bayer_filter(example_image, bayer_filter_tile_rggb)
bayer_filtered_image = tf.expand_dims(bayer_filtered_image, axis = 0)

write_image(bayer_filtered_image, os.path.join(output_dir, 'bayer_filtered_image.png'))



# okay, have a bayer filtered image.
# here's how I'd demosaic, just doing linear interpolation
# once I figure this out, can basically do what i did but learn the weights (and maybe expand from 3x3 to 5x5 kernels

# using 4 kernels:
#
# [[1 2],
#  [3 4]]

#shape is tile_y, tile_x, kernel_x, kernel_y

test_kernel = tf.cast([
        [ [0, 0, 0], [0, 0, 0], [0, 0, 0] ],
        [ [0, 0, 0], [1, 1, 1], [0, 0, 0] ],
        [ [0, 0, 0], [0, 0, 0], [0, 0, 0] ]
    ], dtype = tf.float32)

#test_kernel = tf.cast([
#        [ [1, 1, 1]],
#    ], dtype = tf.float32)

test_kernel = tf.expand_dims(test_kernel, axis = -2)


#demosaic_image = apply_demosaic_filter(bayer_filtered_image, demosaic_kernels_rggb)

#do you even need the bayer filter?
demosaic_image = apply_demosaic_filter(tf.expand_dims(example_image, axis = 0), demosaic_kernels_rggb)

write_image(demosaic_image, os.path.join(output_dir, 'demosaic_image.png'))

write_image(demosaic_filters_to_image(demosaic_kernels_rggb), os.path.join(output_dir, 'demosaic_filters.png'))
        
    



