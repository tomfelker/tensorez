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
from tensorstax_util import *
from tensorstax_model import *

output_dir = os.path.join('output', 'experiments')

tf.enable_eager_execution()

example_image = read_image(os.path.join('misc_images', 'saturn20131017.jpg'))


shrink_by = 8
size_multiple_of = 2 * 3
example_image = tf.image.resizeaverage_image = tf.image.resize(example_image, (example_image.shape[-3] // shrink_by // size_multiple_of * size_multiple_of, example_image.shape[-2] // shrink_by // size_multiple_of * size_multiple_of), method = tf.image.ResizeMethod.BICUBIC)

write_image(example_image, os.path.join(output_dir, 'example_image.png'))

# shape is (height, width, channel)
bayer_filter_tile = tf.cast([
        [ [1, 0, 0], [0, 1, 0] ],
        [ [0, 1, 0], [0, 0, 1] ]
    ], dtype = tf.float32)

bayer_height = bayer_filter_tile.shape[-3]
bayer_width = bayer_filter_tile.shape[-2]

# hmm, I wonder if there's a better way...
bayer_filter_full = tf.tile(bayer_filter_tile, multiples = (example_image.shape[-3] // bayer_filter_tile.shape[-3], example_image.shape[-2] // bayer_filter_tile.shape[-2], 1))

bayer_filtered_image = example_image * bayer_filter_full

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

demosaic_r = [
    [ [ 0,   0, .25], [0, .25, 0], [0,   0, .25] ],
    [ [ 0, .25,   0], [1,   0, 0], [0, .25,   0] ],
    [ [ 0,   0, .25], [0, .25, 0], [0,   0, .25] ]]

demosaic_g_upper_right = [
    [ [  0, 0, 0], [0, 0, .5], [ 0, 0, 0] ],
    [ [ .5, 0, 0], [0, 1,  0], [.5, 0, 0] ],
    [ [  0, 0, 0], [0, 0, .5], [ 0, 0, 0] ]]

demosaic_g_lower_left = [
    [ [ 0, 0,  0], [.5, 0, 0], [0, 0,  0] ],
    [ [ 0, 0, .5], [ 0, 1, 0], [0, 0, .5] ],
    [ [ 0, 0,  0], [.5, 0, 0], [0, 0,  0] ]]

demosaic_b = [
    [ [ .25,   0, 0], [0, .25, 0], [.25,   0, 0] ],
    [ [   0, .25, 0], [0,   0, 1], [  0, .25, 0] ],
    [ [ .25,   0, 0], [0, .25, 0], [.25,   0, 0] ]]
        
demosaic_kernels = tf.cast([
        [ demosaic_r,            demosaic_g_upper_right ],
        [ demosaic_g_lower_left, demosaic_b ]
    ], dtype = tf.float32)

demosaic_kernels = tf.expand_dims(demosaic_kernels, axis = -2)

test_kernel = tf.cast([
        [ [0, 0, 0], [0, 0, 0], [0, 0, 0] ],
        [ [0, 0, 0], [1, 1, 1], [0, 0, 0] ],
        [ [0, 0, 0], [0, 0, 0], [0, 0, 0] ]
    ], dtype = tf.float32)

#test_kernel = tf.cast([
#        [ [1, 1, 1]],
#    ], dtype = tf.float32)

test_kernel = tf.expand_dims(test_kernel, axis = -2)

col_subimages = []
for tile_y in range(0, bayer_height):
    row_subimages = []
    for tile_x in range(0, bayer_width):
        demosaic_kernel_for_tile_pos = demosaic_kernels[tile_y, tile_x, ...]
        #demosaic_kernel_for_tile_pos = test_kernel

        # needing to shift by the center of the kernel seems like a bug...
        shifted_image = tf.roll(bayer_filtered_image, shift = (-tile_y + (demosaic_kernel_for_tile_pos.shape[0].value // 2), -tile_x + (demosaic_kernel_for_tile_pos.shape[1].value // 2)), axis = (-3, -2))
        
        subimage_for_tile_pos = tf.nn.conv2d(shifted_image, demosaic_kernel_for_tile_pos, strides = (bayer_height, bayer_width), padding = 'SAME')

        write_image(subimage_for_tile_pos, os.path.join(output_dir, 'subimage_for_tile_x_{}_y_{}.png'.format(tile_x, tile_y)))

        # and now for a bunch of voodoo magic with indices to recombine this monstrosity...
        row_subimages.append(subimage_for_tile_pos)
    row_subimages = tf.stack(row_subimages, axis = -2)    
    row_subimages = tf.reshape(row_subimages, shape = (bayer_filtered_image.shape[-3] // bayer_height, bayer_filtered_image.shape[-2], bayer_filtered_image.shape[-1]))
    write_image(row_subimages, os.path.join(output_dir, 'subimage_for_tile_y_{}.png'.format(tile_y)))
    col_subimages.append(row_subimages)
col_subimages = tf.stack(col_subimages, axis = -3)
col_subimages = tf.reshape(col_subimages, shape = (bayer_filtered_image.shape[-3], bayer_filtered_image.shape[-2], bayer_filtered_image.shape[-1]))

demosaic_image = col_subimages

write_image(demosaic_image, os.path.join(output_dir, 'demosaic_image.png'))

print("demosaic_kernels.shape {}".format(demosaic_kernels.shape))
        
    



