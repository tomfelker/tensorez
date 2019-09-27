import numpy as np
import tensorflow as tf


tf.enable_eager_execution()

# image has shape (batch, height, width, channels)
# bayer_filter has shape (height, width, channels) and is generally 2x2  [r,g],[g,b], and tiles across the image
def apply_bayer_filter(image, bayer_filter_tile):
    bayer_filter_full = tf.tile(bayer_filter_tile, multiples = (image.shape[-3] // bayer_filter_tile.shape[-3], image.shape[-2] // bayer_filter_tile.shape[-2], 1))
    bayer_filtered_image = image * bayer_filter_full
    return bayer_filtered_image

#bayer_filtered_image has shape (batch, height, width, channels)
#demosaic_kernels has shape (bayer_height, bayer_width, kernel_height, kernel_width, 1, channels)
def apply_demosaic_filter(bayer_filtered_image, demosaic_kernels):
    bayer_height = demosaic_kernels.shape[-6]
    bayer_width = demosaic_kernels.shape[-5]
    kernel_height = demosaic_kernels.shape[-4].value
    kernel_width = demosaic_kernels.shape[-3].value
    
    col_subimages = []
    for tile_y in range(0, bayer_height):
        row_subimages = []
        for tile_x in range(0, bayer_width):
            demosaic_kernel_for_tile_pos = demosaic_kernels[..., tile_y, tile_x, :, :, :, :]

            # shift the image, so the strides hit the appropriate pixels, all from the correct part of the tile pattern
            # also needing to shift by the center of the kernel seems like a tf bug...
            shifted_image = tf.roll(bayer_filtered_image, shift = (-tile_y + (kernel_height // 2), -tile_x + (kernel_width // 2)), axis = (-3, -2))
            
            subimage_for_tile_pos = tf.nn.conv2d(shifted_image, demosaic_kernel_for_tile_pos, strides = (bayer_height, bayer_width), padding = 'SAME')
            
            # and now for a bunch of voodoo magic with indices to recombine this monstrosity...
            row_subimages.append(subimage_for_tile_pos)
        row_subimages = tf.stack(row_subimages, axis = -2)    
        row_subimages = tf.reshape(row_subimages, shape = (bayer_filtered_image.shape[-4], bayer_filtered_image.shape[-3] // bayer_height, bayer_filtered_image.shape[-2], bayer_filtered_image.shape[-1]))
        col_subimages.append(row_subimages)
    col_subimages = tf.stack(col_subimages, axis = -3)
    col_subimages = tf.reshape(col_subimages, shape = (bayer_filtered_image.shape[-4], bayer_filtered_image.shape[-3], bayer_filtered_image.shape[-2], bayer_filtered_image.shape[-1]))
    return col_subimages


# And now some common kernels, for simulation and testing:
# though really we want to learn these...

# shape is (height, width, channel)
bayer_filter_tile_rggb = tf.cast([
        [ [1, 0, 0], [0, 1, 0] ],
        [ [0, 1, 0], [0, 0, 1] ]
    ], dtype = tf.float32)


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
        
demosaic_kernels_rggb = tf.cast([
        [ demosaic_r,            demosaic_g_upper_right ],
        [ demosaic_g_lower_left, demosaic_b ]
    ], dtype = tf.float32)

demosaic_kernels_rggb = tf.expand_dims(demosaic_kernels_rggb, axis = -2)

