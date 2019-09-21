import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg
import os
import glob

# this would be so cool, but not supported in windows, and deprecated anyway...
#import tensorflow.contrib.ffmpeg

# hmm, isn't there some way to not specify it?
data_path = os.path.join('data', 'saturn_bright_mvi_6902')
file_glob = os.path.join(data_path, '????????.png')

#observed_image_int = tf.image.decode_image(observed_image_bytes, channels = num_channels)
#observed_image = tfg.image.color_space.srgb.to_linear(observed_image_int)

#estimated_image_sum = tf.Variable(tf.zeros(shape = (image_width, image_height, num_channels)))

#next_estimated_image_sum = tf.add(estimated_image_sum, observed_image)

#update_op = tf.assign(estimated_image_sum, next_estimated_image_sum)


tf.enable_eager_execution()


def read_image(filename):
    print("Reading", filename)
    image_bytes = tf.io.read_file(filename)
    image_srgb_int = tf.io.decode_image(image_bytes)
    image_srgb = tf.cast(image_srgb_int, tf.float32) / 255.0
    image = tfg.image.color_space.linear_rgb.from_srgb(image_srgb)
    return image

def write_image(image, filename):
    print("Writing", filename)
    image_srgb = tfg.image.color_space.srgb.from_linear_rgb(image) * 255.0  #hmm, correct rounding?
    image_srgb_int = tf.cast(image_srgb, tf.uint8)
    image_bytes = tf.image.encode_png(image_srgb_int)
    tf.io.write_file(filename, image_bytes)

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

def center_image(image, pad = 0):
    spatial_dims = get_spatial_dims()
    
    if pad is not 0:
        paddings = tf.constant([[pad, pad], [pad, pad], [0, 0]])
        image = tf.pad(image, paddings)

    com = center_of_mass(image)

    shift = tf.squeeze(com)
    shift = tf.cast(shift, tf.int32)
    shift = shift * -1
    print("shift:", shift)
    image = tf.roll(image, shift = shift, axis = spatial_dims)

    return image

    
    
        
        



estimated_image_sum = None
num_images = 0
for filename in glob.glob(file_glob):
    num_images += 1
    observed_image = read_image(filename)

    observed_image = center_image(observed_image)


    if estimated_image_sum is None:
        estimated_image_sum = observed_image
    else:
        estimated_image_sum += observed_image    

    #break
                
estimated_image = estimated_image_sum / num_images

print(estimated_image.dtype)

write_image(estimated_image, os.path.join(data_path, 'sum.png'))
print("Cool");

    
