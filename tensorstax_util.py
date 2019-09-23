import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg


def read_image(filename, to_float = True, srgb_to_linear = True):
    print("Reading", filename)
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image)
    if to_float:
        image = tf.cast(image, tf.float32) / 255.0
        if srgb_to_linear:
            image = tfg.image.color_space.linear_rgb.from_srgb(image)
    return image

def write_image(image, filename):
    print("Writing", filename)
    image_srgb = tfg.image.color_space.srgb.from_linear_rgb(image) * 255.0  #hmm, correct rounding?
    image_srgb_int = tf.cast(image_srgb, tf.uint8)
    image_bytes = tf.image.encode_png(image_srgb_int)
    tf.io.write_file(filename, image_bytes)

def write_sequential_image(image, basename, sequence_num):
    write_image(image, basename + "_latest.png")
    write_image(image, basename + "_{:08d}.png".format(sequence_num))

# return an array of which dimensions are x, y, etc., with channels being -1st dim
def get_spatial_dims(num_spatial_dims = 2):
    spatial_dims = []
    for spatial_dim_index in range(0, num_spatial_dims):
        spatial_dims.append(-2 - spatial_dim_index)
    return spatial_dims

def center_of_mass(image, num_spatial_dims = 2, collapse_channels = True):
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

def center_image(image, pad = 0):
    image = pad_image(image, pad)
    
    spatial_dims = get_spatial_dims()    

    com = center_of_mass(image)

    shift = tf.squeeze(com, axis = -1)    
    shift = tf.cast(shift, tf.int32)

    #theory - only shifting by multiples of 2 may help avoid artifacts do to sensor debayering
    shift = tf.bitwise.bitwise_and(shift, -2)
    
    shift = shift * -1
    #print("shift:", shift)
    image = tf.roll(image, shift = shift, axis = spatial_dims)

    return image


def center_image_per_channel(image, pad = 0):
    channel_images = tf.split(image, image.shape[-1], axis = -1)
    
    centered_channel_images = []
    for channel_image in channel_images:
        centered_channel_image = center_image(channel_image, pad)
        centered_channel_images.append(centered_channel_image)

    centered_channel_images = tf.concat(centered_channel_images, axis = -1)
    return centered_channel_images

