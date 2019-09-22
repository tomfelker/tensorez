import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg
import os
import glob
from tensorstax_util import *
from tensorstax_model import *

# this would be so cool, but not supported in windows, and deprecated anyway...
#import tensorflow.contrib.ffmpeg




psf_size = 64

psf_training_rate = .01
psf_training_steps = 100

estimate_training_rate = .1
estimate_training_steps = 100

estimate_update_rate = .1

#data_path = os.path.join('data', 'saturn_bright_mvi_6902')
#data_path = os.path.join('data', 'jupiter_mvi_6906')
data_path = os.path.join('obd', 'data','epsilon_lyrae')

file_glob = os.path.join(data_path, '????????.png')

tf.enable_eager_execution()



#ema means exponential moving average
estimated_image_ema = None
#estimated_sensor_bias_image = None
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
    if estimated_image_ema is None:
        estimated_image_ema = tf.Variable(observed_image)

    #if estimated_sensor_bias_image is None:
    #    estimated_sensor_bias_image = tf.Variable(tf.zeros_like(observed_image))

    
    #observed_image -= estimated_sensor_bias_image
    

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
        
        psf_guess = tf.random.uniform(psf_shape)
        #psf_guess = tf.ones(psf_shape)
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
            predicted_observed_image = tf.nn.conv2d(estimated_image_ema, psf_guess, padding = 'SAME')
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

    # normalize psf
    # though, for some reason, doing it in above loop makes it fail to converge...
    tf.assign(psf_guess, psf_guess / tf.reduce_sum(psf_guess))

    write_image(tf.squeeze(predicted_observed_image), "predicted_observed_image.png")
    write_image(tf.squeeze(psf_guess) / tf.reduce_max(psf_guess), "psf_guess.png")

    # and now update the estimate
    # hmm, this is basically copypasta of the above

    estimated_image = tf.Variable(estimated_image_ema)
    
    estimate_optimizer = tf.train.AdamOptimizer(estimate_training_rate)
    for train_step in range(0, estimate_training_steps):
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(estimated_image)
            #tape.watch(estimated_sensor_bias_image)
            predicted_observed_image = tf.nn.conv2d(estimated_image, psf_guess, padding = 'SAME')
            ## hmm, this seems deprecated, but what is the replacement?  I could write it myself, but why?
            loss = tf.losses.mean_squared_error(observed_image, predicted_observed_image)
            print("Estimated image training step", train_step, "Loss:", loss.numpy())
        # don't really need to be this manual, could use optimizer.compute_gradients
        d_estimated_image_d_loss = tape.gradient(loss, estimated_image)
        #d_estimated_sensor_bias_image_d_loss = tape.gradient(loss, estimated_sensor_bias_image)
        del tape
        #print("d_psf_guess_d_loss:", d_psf_guess_d_loss)
        estimate_optimizer.apply_gradients([
            (d_estimated_image_d_loss, estimated_image),
            #(d_estimated_sensor_bias_image_d_loss, estimated_sensor_bias_image)
            ])
        tf.assign(estimated_image, tf.math.maximum(estimated_image, 0))
        #psf_guess = tf.(psf_guess, 0)
        #hmm, constrain positivity?
        # above doesn't work...

    tf.assign(estimated_image_ema, estimated_image_ema * (1 - estimate_update_rate) + estimated_image * estimate_update_rate)
     
    write_image(tf.squeeze(estimated_image/tf.reduce_max(estimated_image)), "estimated_image.png")
    write_image(tf.squeeze(estimated_image_ema/tf.reduce_max(estimated_image_ema)), "estimated_image_ema.png")
    write_image(center_image_per_channel(tf.squeeze(estimated_image/tf.reduce_max(estimated_image))), "estimated_image_centered.png")
    #write_image(tf.squeeze(estimated_sensor_bias_image/tf.reduce_max(estimated_sensor_bias_image)), "estimated_sensor_bias_image.png")
    #break
                
estimated_image_avg = estimated_image_sum / num_images
estimated_image_avg = tf.squeeze(estimated_image_avg)
estimated_image_avg = center_image_per_channel(estimated_image_avg)

write_image(estimated_image_avg, os.path.join(data_path, 'avg.png'))
print("Cool");

    
