import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg
import fnmatch
import os
import math

from tensorez.bayer import *

from PIL import Image
try:
    import rawpy
except:
    print("Failed to import RawPY, needed for raw (CR2, etc.) support...")

def map_along_tensor_axis(func, tensor, axis, keepdims = False):
    if keepdims:
        return tf.concat(list(map((lambda unstacked: func(tf.expand_dims(unstacked, axis = axis))), tf.unstack(tensor, axis = axis))), axis = axis)
    else:
        return tf.stack(list(map(func, tf.unstack(tensor, axis = axis))), axis = axis)


# I have no idea why this doesn't seem to come with TensorFlow...
def gaussian(x, mean = 0, variance = 1):
    sigma = math.sqrt(variance)
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * tf.exp(-0.5 * tf.square((x - mean) / sigma))

def gaussian_psf(size, standard_deviation = .1):    
    
    coords = tf.linspace(-0.5, 0.5, size)
    # for gaussian, really we could just evaluate along the line and multiply, but
    # this way lets us try other radially-symmetric functions:
    coords_x = tf.expand_dims(coords, axis = -2)
    coords_y = tf.expand_dims(coords, axis = -1)
    coords_r = tf.sqrt(tf.square(coords_x) + tf.square(coords_y))
    
    psf = gaussian(coords_r, variance = standard_deviation * standard_deviation)
    #psf = 0.5 - coords_r

    psf = psf / tf.reduce_sum(psf)
    psf = tf.expand_dims(psf, axis = -1)
    return psf


def promote_to_three_channels(image):
    if image.shape.ndims == 2:
        image = tf.expand_dims(image, axis = -1)
    if image.shape.ndims == 4:
        image = tf.squeeze(image, axis = -4)
    if image.shape[-1] == 1:
        image = tf.concat([image, image, image], axis = -1)
    return image

def crop_image(image, crop, crop_align):
    if crop is not None:
        width, height = crop
        x = (image.shape[-2] - width) // 2
        y = (image.shape[-3] - height) // 2
        x = (x // crop_align) * crop_align
        y = (y // crop_align) * crop_align
        return image[..., y : y + height, x : x + width, :]
    return image

def read_image(filename, to_float = True, srgb_to_linear = True, crop = None, crop_align = 2, color_balance = True, demosaic = True):
    print("Reading", filename)
    if fnmatch.fnmatch(filename, '*.tif') or fnmatch.fnmatch(filename, '*.tiff'):
        image = Image.open(filename)        
        image = np.array(image)
        image = crop_image(image, crop = crop, crop_align = crop_align)
    if fnmatch.fnmatch(filename, '*.cr2'):
        # For raw files, we will read them into a full color image, but with the bayer pattern... this way,
        # we can easily handle whatever bayer pattern

        with rawpy.imread(filename) as raw:
            # Note: there seems to be a bug where if you leave the image in numpy-land and let raw go out of scope, it silently crashes.
            # Copying to tensorflow seems to work around it.  Making a numpy copy to be doubly sure.
            image = np.copy(raw.raw_image_visible)
            image = tf.expand_dims(image, axis = -1)
            image = tf.expand_dims(image, axis = 0)
            image = crop_image(image, crop = crop, crop_align = crop_align)
            # todo: color balance?
            # todo: black level?
            # todo: curve?
            # todo: warn about weird raw settings
            # todo, why exactly 14 bits?  just my camera?
            if to_float:
                image = tf.cast(image, tf.float32)
                
                if ((color_balance or demosaic) and (
                    chr(raw.color_desc[raw.raw_pattern[0,0]]) != 'R' or
                    chr(raw.color_desc[raw.raw_pattern[0,1]]) != 'G' or
                    chr(raw.color_desc[raw.raw_pattern[1,0]]) != 'G' or
                    chr(raw.color_desc[raw.raw_pattern[1,1]]) != 'B')):
                        print("Warning: raw file has weird bayer pattern {}".format(raw.color_desc))
                
                if color_balance:

                    scale_tile = tf.convert_to_tensor([
                        [raw.daylight_whitebalance[0], raw.daylight_whitebalance[1]],
                        [raw.daylight_whitebalance[1], raw.daylight_whitebalance[2]]
                        ], dtype = tf.float32)
                    black_tile = tf.convert_to_tensor([
                        [raw.black_level_per_channel[raw.raw_pattern[0,0]], raw.black_level_per_channel[raw.raw_pattern[0,1]]],
                        [raw.black_level_per_channel[raw.raw_pattern[1,0]], raw.black_level_per_channel[raw.raw_pattern[1,1]]]                        
                        ], dtype = tf.float32)

                    scale_tile /= 16383.0

                    scale = tf.reshape(tf.tile(scale_tile, multiples = (image.shape[-3] // 2, image.shape[-2] // 2)), image.shape)
                    black = tf.reshape(tf.tile(black_tile, multiples = (image.shape[-3] // 2, image.shape[-2] // 2)), image.shape)

                    image = (image - black) * scale

                if demosaic:
                    image = apply_demosaic_filter(image, demosaic_kernels_rggb)

                           
    else:
        image = tf.io.read_file(filename)
        image = tf.io.decode_image(image)
        image = tf.expand_dims(image, axis = -4)
        image = crop_image(image, crop = crop, crop_align = crop_align)
        if to_float:
            image = tf.cast(image, tf.float32) / 255.0
            if srgb_to_linear:
                image = tfg.image.color_space.linear_rgb.from_srgb(image)
    return image

def write_image(image, filename, normalize = False, saturate = True):
    print("Writing", filename)
    if normalize:
        image = image / tf.reduce_max(image)
    if saturate:
        image = tf.minimum(1.0, image)
    image = promote_to_three_channels(image)
    image_srgb = tfg.image.color_space.srgb.from_linear_rgb(image) * 255.0  #hmm, correct rounding?
    image_srgb_int = tf.cast(image_srgb, tf.uint8)
    image_bytes = tf.image.encode_png(image_srgb_int)
    tf.io.write_file(filename, image_bytes)

def write_sequential_image(image, path, name, sequence_num, extension = 'png', **write_image_args):
    name_with_sequence = "{}_{:08d}.{}".format(name, sequence_num, extension)
    name_with_latest = "{}_latest.{}".format(name, extension)

    if sequence_num == 0:
        history_path = os.path.join(path, 'initial', name_with_sequence)
    else:
        history_path = os.path.join(path, 'history', name, name_with_sequence)
        
    temp_path = os.path.join(path, name_with_sequence)
    latest_path = os.path.join(path, name_with_latest)
    
    write_image(image, history_path, **write_image_args)
    try:
        # hard links on windows!  Let's hope it doesn't all come crashing down... but if we do it right, it's atomic and we only wrote the file once.
        # would be best on linux, where we could link and replace in one atomic step without the temp name, but python doesn't expose this, and windows probably can't do it
        os.link(history_path, temp_path)
        os.replace(temp_path, latest_path)
        
        # and let's copy it up one also...
        temp_path = os.path.join(path, '..', name_with_sequence)
        latest_path = os.path.join(path, '..', name_with_latest)
        os.link(history_path, temp_path)
        os.replace(temp_path, latest_path)
    except Exception as e:
        # sometimes this gets access denied, because windows is lame and its filesystem won't let you be atomic
        print("Problem when moving files: {}".format(e))
    

# return an array of which dimensions are x, y, etc., with channels being -1st dim
def get_spatial_dims(num_spatial_dims = 2):
    spatial_dims = []
    for spatial_dim_index in range(0, num_spatial_dims):
        spatial_dims.append(-2 - spatial_dim_index)
    return spatial_dims

def center_of_mass(image, num_spatial_dims = 2, collapse_channels = True):
    # todo: fix this - it's too complicated, and doesn't work batchwise...
    if image.shape.ndims == 4:
        image = tf.squeeze(image, axis = -4)
    
    #print("image.shape:", image.shape)

    spatial_dims = get_spatial_dims(num_spatial_dims)

    #print("spatial_dims:", spatial_dims)

    if collapse_channels:
        image = tf.reduce_sum(image, axis = -1, keepdims = True)
    
    total_mass = tf.reduce_sum(image, axis = spatial_dims, keepdims = True)        
    #print("total_mass:", total_mass)
    
    ret = None
    for dim in spatial_dims:
        #print("Evaluating CoM in dim", dim)
        dim_size = image.shape[dim]
        #print("which is of size", dim_size)
        multiplier = tf.linspace(-dim_size.value / 2.0, dim_size.value / 2.0, dim_size)

        multiplier_shape = []
        for sum_dim in range(0, tf.rank(image)):
            multiplier_shape.append(1)
        multiplier_shape[dim] = dim_size        
        #print("Multiplier shape:", multiplier_shape)
        
        multiplier_shaped = tf.reshape(multiplier, multiplier_shape)
        moments = tf.multiply(image, multiplier_shaped)
        
        com_in_dim = tf.reduce_sum(moments, axis = spatial_dims, keepdims = True) / total_mass
        #print('com_in_dim.shape', com_in_dim.shape)
        com_in_dim = tf.squeeze(com_in_dim, axis = spatial_dims)
        com_in_dim = tf.stack([com_in_dim], axis = -2)        
        #print('com_in_dim.shape', com_in_dim.shape)

        if ret is None:
            ret = com_in_dim
        else:
            ret = tf.concat([ret, com_in_dim], axis = -2)
    #print(ret)
    return ret

def pad_image(image, pad):
    if pad is not 0:
        paddings = tf.constant([[pad, pad], [pad, pad], [0, 0]])
        image = tf.pad(image, paddings)
    return image

def center_image(image, pad = 0, only_even_shifts = False):
    image = pad_image(image, pad)
    
    spatial_dims = get_spatial_dims()    

    com = center_of_mass(image)

    shift = tf.squeeze(com, axis = -1)    
    shift = tf.cast(shift, tf.int32)

    #theory - only shifting by multiples of 2 may help avoid artifacts do to sensor debayering
    if only_even_shifts:
        shift = tf.bitwise.bitwise_and(shift, -2)
    
    shift = shift * -1
    #print("shift:", shift)
    shift_axis = spatial_dims
    image = tf.roll(image, shift = shift, axis = shift_axis)

    return image, shift, shift_axis


def center_image_per_channel(image, pad = 0, **kwargs):
    return map_along_tensor_axis((lambda image: center_image(image, **kwargs)[0]), image, axis = -1, keepdims = True)
    

def vector_to_graph(v, ysize = None, line_thickness = 1):
    if ysize is None:
        ysize = v.shape[0]
    
    lin = tf.linspace(1.0, 0.0, ysize);
    lin = tf.expand_dims(lin, axis = 1)
    v = tf.expand_dims(v, axis = 0)
    distances_to_line = tf.math.abs(v - lin) * tf.cast(ysize, tf.float32);
    return tf.maximum(0, tf.minimum(1, (line_thickness / 2) - distances_to_line + 1))

def adc_function_to_graph(adc, **kwargs):
    channels = tf.unstack(adc, axis = -1)
    channels = list(map((lambda channel: vector_to_graph(channel, **kwargs)), channels))
    return tf.stack(channels, axis = -1)

def center_images(images, **kwargs):
    print("centering images, shape {}".format(images.shape))
    return map_along_tensor_axis((lambda image: center_image(image, **kwargs)[0]), images, 0)

    

    
