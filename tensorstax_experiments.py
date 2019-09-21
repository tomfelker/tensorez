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

estimated_image_sum = None
num_images = 0
for filename in glob.glob(file_glob):
    num_images += 1
    observed_image = read_image(filename)

    if estimated_image_sum is None:
        estimated_image_sum = observed_image
    else:
        estimated_image_sum += observed_image    

estimated_image = estimated_image_sum / num_images

print(estimated_image.dtype)

write_image(estimated_image, os.path.join(data_path, 'sum.png'))
print("Cool");

    
