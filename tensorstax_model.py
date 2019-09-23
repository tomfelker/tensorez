import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg

from tensorstax_util import *

# Okay, this model is philosophically weird, in that it doesn't have any outputs.
# We're not trying to get a quick forward pass, and minimize its error vs. some known data.
# Instead, the entire purpose of the model is that after training, some of its weights will be interesting,
# because for example estimated_image will be a pretty picture of our belief about the night sky.
#
# It will still have inputs, however, which are the observed data (which are actually at the _end_ of the processing).
# Largely this is just to take advantage of all the existing infrastrucutre for batching and so forth.
#
# Another oddity: some of the weights, namely the point-spread functions, are relevant only to a single training example.
# So if we get a new batch of data, we should clear the point-spread functions.  And furthermore, to actually learn the
# point spread functions, we must process the same batch many times.  This is quite different from a normal model where
# all of the weights are meant to generalize, and so are trained using (random subsets of) all examples.
#
# Consequently, we will need a custom training loop.  (Or lots of callback hackery...)
# And perhaps the only purpose of making this a Keras model is for the handy storing of trainable variables...


class TensorstaxModel(tf.keras.Model):


    def __init__(self, psf_size = 32, model_adc = True):
        super(TensorstaxModel, self).__init__()
        self.psf_size = psf_size
        self.model_adc = model_adc

    def build(self, input_shape):
        (self.batch_size, self.width, self.height, self.channels) = input_shape
        print("Building model: batch_size {}, width {}, height {}, channels {}".format(self.batch_size, self.width, self.height, self.channels))

        # not just psf, but any variables that should be trained in that step, optimizing per-example beliefs, but holding our long-term belief about the world constant.
        self.psf_training_vars = []

        # not just the image, but any variables that should be trained in the second step, holding per-example beliefs constant, but optimizing our long-term belief about the world
        self.image_training_vars = []

        self.estimated_image = tf.Variable(tf.zeros((self.width, self.height, self.channels)))
        self.image_training_vars.append(self.estimated_image)
        self.have_initial_estimate = False
        

        self.point_spread_functions = tf.Variable(self.default_point_spread_functions())
        self.psf_training_vars.append(self.point_spread_functions)

        if self.model_adc:
            adc_guess = tf.linspace(0.0, 1.0, 256)           
            
            # for some reason, from_srgb really wants 3 - is it doing more than just the gamma curve?
            adc_guess = tf.expand_dims(adc_guess, axis = -1)
            adc_guess = tf.broadcast_to(adc_guess, shape = (256, 3))

            self.adc_function_linear = adc_guess
            
            adc_guess = tfg.image.color_space.linear_rgb.from_srgb(adc_guess)
            self.adc_function_srgb = adc_guess
        
            #print("ADC:")
            #print(adc_guess.numpy())

            #just one channel?
            #adc_guess = adc_guess[..., 0]
            
            self.adc_function = tf.Variable(adc_guess)
            self.image_training_vars.append(self.adc_function)

    def default_point_spread_functions(self):
        psfs_shape = (self.batch_size, self.psf_size, self.psf_size, 1, self.channels)
        
        psfs = tf.random.uniform(psfs_shape)
        psfs = psfs / tf.reduce_sum(psfs, axis = (-4, -3, -1), keepdims = True)

        #psfs = tf.zeros(psfs_shape)
        
        return psfs


    def call(self, observed_images):
        
        observed_images = self.apply_adc_function(observed_images)
        
        if not self.have_initial_estimate:
            print("Averaging images for initial estimate:")            
            tf.assign(self.estimated_image, tf.reduce_mean(observed_images, axis = 0))
            self.have_initial_estimate = True

        estimated_image_extra_dim = tf.expand_dims(self.estimated_image, axis = 0)
        
        predicted_observed_images = []
        for example_index in range(0, self.batch_size):            
            predicted_observed_image = tf.nn.conv2d(estimated_image_extra_dim, self.point_spread_functions[example_index, ...], padding = 'SAME')
            predicted_observed_image = tf.squeeze(predicted_observed_image, axis = 0)
            predicted_observed_images.append(predicted_observed_image)
        predicted_observed_images = tf.stack(predicted_observed_images, axis = 0)

        self.add_loss(tf.losses.mean_squared_error(observed_images, predicted_observed_images))

    def apply_psf_physicality_constraints(self):
        # apply physicality constraints, PSF must be positive and normal
        tf.assign(self.point_spread_functions, tf.maximum(0, self.point_spread_functions))
        # hmm, why doesn't this help?  maybe it got NaNs when PSFs go to zero with initial training overshoot?
        #tf.assign(model.point_spread_functions, model.point_spread_functions / tf.reduce_sum(model.point_spread_functions, axis = (-4, -3), keepdims = True))

    def apply_image_physicality_constraints(self):
        # apply physicality constraints, image must be positive
        tf.assign(self.estimated_image, tf.maximum(0, self.estimated_image))

        if self.model_adc:
            adc_min = tf.reduce_min(self.adc_function, axis = 0, keepdims = True)
            print("ADC min: {}".format(adc_min.numpy()))
            tf.assign(self.adc_function, self.adc_function - adc_min)
            
            adc_max = tf.reduce_max(self.adc_function, axis = 0, keepdims = True)
            print("ADC max: {}".format(adc_max.numpy()))
            tf.assign(self.adc_function, self.adc_function / adc_max)


    def apply_adc_function(self, observed_images):
        if self.model_adc:
            observed_images = tf.cast(observed_images, tf.int32) # just to avoid error, it can't use uint8
            # just one channel would be:
            #observed_images = tf.gather(self.adc_function, observed_images)

            # there's gotta be a better way to do this...            
            observed_channels = tf.unstack(observed_images, axis = -1)            
            observed_images = []
            for channel_index in range(0, self.channels):
                adc_channel = self.adc_function[..., channel_index]
                observed_images.append(tf.gather(adc_channel, observed_channels[channel_index]))
            observed_images = tf.stack(observed_images, axis = -1)
                                       
        return observed_images
            
    def get_psf_examples(self, num_examples = 5):
        psf_examples = self.point_spread_functions[0:num_examples, :, :, 0, :]
        psf_examples = psf_examples / tf.reduce_max(psf_examples, axis = (-3, -2, -1), keepdims = True)
        psf_example = tf.reshape(psf_examples, (psf_examples.shape[-3] * psf_examples.shape[-4], psf_examples.shape[-2], psf_examples.shape[-1]))
        return psf_example

    def print_adc_stats(self):
        if self.model_adc:
            print("ADC function:")
            print(self.adc_function.numpy())

            print("ADC MSE vs linear: {}".format(tf.losses.mean_squared_error(self.adc_function, self.adc_function_linear).numpy()))
            print("ADC MSE vs   sRGB: {}".format(tf.losses.mean_squared_error(self.adc_function, self.adc_function_srgb).numpy()))
