import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg
import os
import glob

# this would be so cool, but not supported in windows, and deprecated anyway...
#import tensorflow.contrib.ffmpeg


psf_size = 32

psf_training_rate = .1
psf_training_steps = 100

estimate_training_rate = .01
estimate_training_steps = 2


#data_path = os.path.join('data', 'saturn_bright_mvi_6902')
#data_path = os.path.join('data', 'jupiter_mvi_6906')
data_path = os.path.join('obd', 'data','epsilon_lyrae')

file_glob = os.path.join(data_path, '????????.png')

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

estimated_image = None
estimated_image_sum = None
num_images = 0
for filename in glob.glob(file_glob):    
    observed_image = read_image(filename)

    observed_image = center_image(observed_image)

    # conv2d requires a batch dimension on our images...
    observed_image = tf.expand_dims(observed_image, axis = 0)

    # Simple summation of the shifted images, as a control:
    num_images += 1
    if estimated_image_sum is None:
        estimated_image_sum = observed_image
    else:
        estimated_image_sum += observed_image    


    # Fancier stuff:
    if estimated_image is None:
        estimated_image = tf.Variable(observed_image)

    # width, height, in_channels, out_channels
    # and in_channels is not what we want... hmm, is that correct?  experimentally yes
    psf_shape = [psf_size, psf_size, 1, observed_image.shape[-1]]
    
    psf_test = False
    if psf_test:
        # width, height, in_channels, out_channels
        # and in_channels is not what we want... hmm, is that correct?  experimentally yes
        psf_guess = np.zeros(psf_shape)
        psf_guess[           0,            0, 0, 0] = 1
        psf_guess[           0, psf_size - 1, 0, 1] = 1
        psf_guess[psf_size - 1,            0, 0, 2] = 1
        psf_guess = tf.cast(psf_guess, tf.float32)
    else:
        
        #psf_guess = tf.random.uniform(psf_shape)
        psf_guess = tf.ones(psf_shape)
        psf_guess *= (2.0 / (psf_size * psf_size))

    psf_guess = tf.Variable(psf_guess)

    # should be ones
    #print("psf_guess sum:", tf.reduce_sum(psf_guess, axis = (0, 1)))

    #okay, first, learning the PSF for this observed image:
    psf_optimizer = tf.train.AdamOptimizer(psf_training_rate)
    for train_step in range(0, psf_training_steps):


        #predicted_observed_image = tf.nn.conv2d(estimated_image, psf_guess, padding = 'SAME')
        #def loss():
        #    # hmm, this seems deprecated, but what is the replacement?  I could write it myself, but why?
        #    return tf.losses.mean_squared_error(observed_image, predicted_observed_image)
        
        with tf.GradientTape() as tape:
            tape.watch(psf_guess)
            predicted_observed_image = tf.nn.conv2d(estimated_image, psf_guess, padding = 'SAME')
            ## hmm, this seems deprecated, but what is the replacement?  I could write it myself, but why?
            loss = tf.losses.mean_squared_error(observed_image, predicted_observed_image)
            print("PSF Training step", train_step, "Loss:", loss.numpy())
        # don't really need to be this manual, could use optimizer.compute_gradients
        d_psf_guess_d_loss = tape.gradient(loss, psf_guess)
        #print("d_psf_guess_d_loss:", d_psf_guess_d_loss)
        psf_optimizer.apply_gradients([(d_psf_guess_d_loss, psf_guess)])
        tf.assign(psf_guess, tf.math.maximum(psf_guess, 0))
        #psf_guess = tf.(psf_guess, 0)
        #hmm, constrain positivity?
        # above doesn't work...

        #optimizer.minimize(loss, var_list=[psf_guess])

    write_image(tf.squeeze(predicted_observed_image), "predicted_observed_image.png")
    write_image(tf.squeeze(psf_guess) / tf.reduce_max(psf_guess), "psf_guess.png")

    # and now update the estimate
    # hmm, this is basically copypasta of the above
    estimate_optimizer = tf.train.AdamOptimizer(estimate_training_rate)
    for train_step in range(0, estimate_training_steps):
        with tf.GradientTape() as tape:
            tape.watch(estimated_image)
            predicted_observed_image = tf.nn.conv2d(estimated_image, psf_guess, padding = 'SAME')
            ## hmm, this seems deprecated, but what is the replacement?  I could write it myself, but why?
            loss = tf.losses.mean_squared_error(observed_image, predicted_observed_image)
            print("Estimated image training step", train_step, "Loss:", loss.numpy())
        # don't really need to be this manual, could use optimizer.compute_gradients
        d_estimated_image_d_loss = tape.gradient(loss, estimated_image)
        #print("d_psf_guess_d_loss:", d_psf_guess_d_loss)
        estimate_optimizer.apply_gradients([(d_estimated_image_d_loss, estimated_image)])
        tf.assign(estimated_image, tf.math.maximum(estimated_image, 0))
        #psf_guess = tf.(psf_guess, 0)
        #hmm, constrain positivity?
        # above doesn't work...
        
    write_image(tf.squeeze(estimated_image/tf.reduce_max(estimated_image)), "estimated_image.png")

    #break
                
estimated_image_avg = estimated_image_sum / num_images
estimated_image_avg = tf.squeeze(estimated_image_avg)
estimated_image_avg = center_image_per_channel(estimated_image_avg)

write_image(estimated_image_avg, os.path.join(data_path, 'avg.png'))
print("Cool");

    
