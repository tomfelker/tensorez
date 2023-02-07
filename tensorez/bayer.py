import numpy as np
import tensorflow as tf

# don't like putting this at module scope, but needed for the hardcoded tensors below
tf.compat.v1.enable_eager_execution()

# image has shape (batch, height, width, channels)
# bayer_filter has shape (height, width, channels) and is generally 2x2  [r,g],[g,b], and tiles across the image
@tf.function
def apply_bayer_filter(image, bayer_filter_tile):
    bayer_filter_full = tf.tile(bayer_filter_tile, multiples = (image.shape[-3] // bayer_filter_tile.shape[-3], image.shape[-2] // bayer_filter_tile.shape[-2], 1))
    bayer_filtered_image = image * bayer_filter_full
    bayer_filtered_image_mono = tf.reduce_sum(bayer_filtered_image, axis = -1, keepdims = True)
    return bayer_filtered_image_mono

#bayer_filtered_image has shape (batch, height, width, channels)
#demosaic_kernels has shape (bayer_height, bayer_width, kernel_height, kernel_width, 1, channels)
@tf.function
def apply_demosaic_filter(bayer_filtered_image, demosaic_kernels):
    bayer_height = demosaic_kernels.shape[-6]
    bayer_width = demosaic_kernels.shape[-5]
    kernel_height = demosaic_kernels.shape[-4]
    kernel_width = demosaic_kernels.shape[-3]

    # using reflect padding is only correct for a kernel size of 2,
    # and odd-sized demosaic kernels would probably at least require uneven padding
    assert bayer_height == 2 and bayer_width == 2
    assert kernel_height % 2 == 1 and kernel_width % 2 == 1
    bayer_filtered_image_padded = tf.pad(
        bayer_filtered_image,
        paddings = [
            [0, 0],
            [kernel_height // 2, kernel_height // 2],
            [kernel_width // 2, kernel_width // 2],
            [0, 0]],
        mode = 'REFLECT')
    
    col_subimages = []
    for tile_y in range(0, bayer_height):
        row_subimages = []
        for tile_x in range(0, bayer_width):
            demosaic_kernel_for_tile_pos = demosaic_kernels[..., tile_y, tile_x, :, :, :, :]

            # todo: as a possible optimization, might be able to use negative explicit pads in conv2d instead of this roll.
            # shift the image, so the strides hit the appropriate pixels, all from the correct part of the tile pattern
            shifted_image = tf.roll(bayer_filtered_image_padded, shift = (-tile_y, -tile_x), axis = (-3, -2))
            
            subimage_for_tile_pos = tf.nn.conv2d(shifted_image, demosaic_kernel_for_tile_pos, strides = (bayer_height, bayer_width), padding = 'VALID')
            
            # and now for a bunch of voodoo magic with indices to recombine this monstrosity...
            row_subimages.append(subimage_for_tile_pos)
        row_subimages = tf.stack(row_subimages, axis = -2)    
        row_subimages = tf.reshape(row_subimages, shape = (bayer_filtered_image.shape[-4], bayer_filtered_image.shape[-3] // bayer_height, bayer_filtered_image.shape[-2], demosaic_kernels.shape[-1]))
        col_subimages.append(row_subimages)
    col_subimages = tf.stack(col_subimages, axis = -3)
    col_subimages = tf.reshape(col_subimages, shape = (bayer_filtered_image.shape[-4], bayer_filtered_image.shape[-3], bayer_filtered_image.shape[-2], demosaic_kernels.shape[-1]))
    return col_subimages

def demosaic_filters_to_image(demosaic_filters):
    #return tf.reshape(demosaic_filters, (demosaic_filters.shape[-4] * demosaic_filters.shape[-6], demosaic_filters.shape[-3] * demosaic_filters.shape[-5], demosaic_filters.shape[-1]))

    image = demosaic_filters
    image = tf.concat(tf.unstack(image), axis = 0)
    image = tf.concat(tf.unstack(image), axis = 0)
    image = tf.squeeze(image, axis = -2)
    return image

# And now some common kernels, for simulation and testing:
# though really we want to learn these...

# shape is (height, width, channel)
bayer_filter_tile_rggb = tf.cast([
        [ [1, 0, 0], [0, 1, 0] ],
        [ [0, 1, 0], [0, 0, 1] ]
    ], dtype = tf.float32)

bayer_filter_tile_grbg = tf.reverse(bayer_filter_tile_rggb, axis = [-2])


demosaic_null = [
    [ [ 0, 0, 0], [0, 0, 0], [0, 0, 0] ],
    [ [ 0, 0, 0], [1, 1, 1], [0, 0, 0] ],
    [ [ 0, 0, 0], [0, 0, 0], [0, 0, 0] ]]

# demosaic_* shape is [filter_height, filter_width, filter_channel], value is how much to look at that channel

# R G
# G B

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
    
# demosaic_kernels_rggb shape will be [height, width, filter_height, filter_width, output_channel?, filter_channel]

demosaic_kernels_rggb = tf.cast([
        [ demosaic_r,            demosaic_g_upper_right ],
        [ demosaic_g_lower_left, demosaic_b ]
    ], dtype = tf.float32)
demosaic_kernels_rggb = tf.expand_dims(demosaic_kernels_rggb, axis = -2)

# G R
# B G
demosaic_kernels_grbg = tf.reverse(demosaic_kernels_rggb, axis = [-3, -5])

demosaic_kernels_null = tf.cast([
        [ demosaic_null, demosaic_null ],
        [ demosaic_null, demosaic_null ]
    ], dtype = tf.float32)
demosaic_kernels_null = tf.expand_dims(demosaic_kernels_null, axis = -2)

