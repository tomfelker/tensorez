import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg

from tensorez.util import *
from tensorez.bayer import *

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


class TensoRezModel(tf.keras.Model):


    def __init__(self, psf_size = 32, model_adc = False, model_noise = False, super_resolution_factor = 2, realign_center_of_mass = True, model_bayer = False, model_demosaic = True, bayer_tile_size = 2, demosaic_filter_size = 3):
        super(TensoRezModel, self).__init__()
        self.psf_size = psf_size
        self.model_adc = model_adc
        self.model_noise = model_noise
        self.super_resolution_factor = super_resolution_factor
        self.realign_center_of_mass = realign_center_of_mass
        self.model_bayer = model_bayer
        self.model_demosaic = model_demosaic
        self.bayer_tile_size = bayer_tile_size
        self.demosaic_filter_size = demosaic_filter_size

        print("Instantiating TensoRez model.")

        # not just psf, but any variables that should be trained in that step, optimizing per-example beliefs, but holding our long-term belief about the world constant.
        self.psf_training_vars = []

        # not just the image, but any variables that should be trained in the second step, holding per-example beliefs constant, but optimizing our long-term belief about the world
        self.image_training_vars = []

        self.already_built = False

        if self.model_adc:
            print("Modeling 8-bit ADC, initial guess of sRGB")
            adc_guess = tf.linspace(0.0, 1.0, 256)           
            
            # for some reason, from_srgb really wants 3 - is it doing more than just the gamma curve?
            adc_guess = tf.expand_dims(adc_guess, axis = -1)
            adc_guess = tf.broadcast_to(adc_guess, shape = (256, 3))

            self.adc_function_linear = adc_guess
            
            adc_guess = tfg.image.color_space.linear_rgb.from_srgb(adc_guess)
            self.adc_function_srgb = adc_guess
                    
            self.adc_function = tf.Variable(adc_guess, name = "adc_function")
            self.image_training_vars.append(self.adc_function)

    def build(self, input_shape):
        if self.already_built:
            return
        self.already_built = True
        
        (self.batch_size, self.width, self.height, self.channels) = input_shape
        print("Building model: batch_size {}, width {}, height {}, channels {}".format(self.batch_size, self.width, self.height, self.channels))

        self.estimated_image = tf.Variable(tf.zeros((1, self.width * self.super_resolution_factor, self.height * self.super_resolution_factor, self.channels)), name = "estimated_image")
        print("estimated_image has shape {}".format(self.estimated_image.shape))
        self.image_training_vars.append(self.estimated_image)
        self.have_initial_estimate = False

        self.point_spread_functions = tf.Variable(self.default_point_spread_functions(), name = "point_spread_functions")
        print("point_spread_functions has shape {}".format(self.point_spread_functions.shape))
        self.psf_training_vars.append(self.point_spread_functions)

        if self.model_noise:
            noise_shape = (self.width, self.height, self.channels)
            
            self.noise_bias = tf.Variable(tf.zeros(noise_shape), name = "noise_bias")
            self.noise_scale = tf.Variable(tf.ones(noise_shape) * (1.0 / 256.0), name = "noise_scale")
            self.noise_image_scale = tf.Variable(tf.ones(noise_shape), name = "noise_image_scale")

            self.image_training_vars.append(self.noise_bias)
            self.image_training_vars.append(self.noise_scale)
            self.image_training_vars.append(self.noise_image_scale)
            
            print("Modeling noise with shape {}".format(noise_shape))
        
        if self.model_bayer:
            #init_bayer_filters = tf.ones(shape = (self.bayer_tile_size, self.bayer_tile_size, self.channels))
            #init_bayer_filters = init_bayer_filters / (self.bayer_tile_size * self.bayer_tile_size)

            # todo: try more to learn these...
            init_bayer_filters = bayer_filter_tile_rggb
            
            self.bayer_filters = tf.Variable(init_bayer_filters, name = "bayer_filters")
            # todo: should this shape have just one channel?  depends if we care to capture crosstalk...

            print("Modeling bayer filters with shape {}".format(self.bayer_filters.shape))
            
            self.image_training_vars.append(self.bayer_filters)

        if self.model_demosaic:
            #init_demosaic_filters = tf.ones(shape = (self.bayer_tile_size, self.bayer_tile_size, self.demosaic_filter_size, self.demosaic_filter_size, 1, self.channels))
            #init_demosaic_filters = init_demosaic_filters / (self.demosaic_filter_size * self.demosaic_filter_size)

            #init_demosaic_filters = demosaic_kernels_rggb

            init_demosaic_filters = demosaic_kernels_null
            
            self.demosaic_filters = tf.Variable(init_demosaic_filters, name = "demosaic_filters")
            print("Modeling demosaic filters with shape {}".format(self.demosaic_filters))
            self.image_training_vars.append(self.demosaic_filters)

    def default_point_spread_functions(self):
        psfs_shape = (self.batch_size, self.psf_size, self.psf_size, 1, self.channels)

        print("Using lame random uniform PSF guess")
        psfs = tf.random.uniform(psfs_shape)
        psfs = psfs / tf.reduce_sum(psfs, axis = (-4, -3, -1), keepdims = True)

        #psfs = tf.zeros(psfs_shape)
        
        return psfs


    def compute_initial_estimate(self, images):
        self.build(images.shape)
        print("Computing initial estimate image:")
        images = self.apply_adc_function(images)
        if self.super_resolution_factor != 1:
            print("Upscaling by {}".format(self.super_resolution_factor))
            images = tf.image.resize(images, (self.width * self.super_resolution_factor, self.height * self.super_resolution_factor), method = tf.image.ResizeMethod.BICUBIC)
            if self.realign_center_of_mass:
                print("Recentering at super-resolution")
                images = center_images(images, only_even_shifts = False)
        print("Averaging")
        average_image = tf.reduce_mean(images, axis = 0, keepdims = True)
        
        self.estimated_image.assign(average_image)
        self.have_initial_estimate = True

    def predict_observed_images(self):
        
        predicted_observed_images = []
        for example_index in range(0, self.batch_size):            
            predicted_observed_image = tf.nn.conv2d(self.estimated_image, self.point_spread_functions[example_index, ...], padding = 'SAME')
            predicted_observed_image = tf.squeeze(predicted_observed_image, axis = 0)
            predicted_observed_images.append(predicted_observed_image)
        predicted_observed_images = tf.stack(predicted_observed_images, axis = 0)

        if self.super_resolution_factor != 1:
            #todo: should I instead just stride the above convolution?  might be equivalent...
            predicted_observed_images = tf.nn.avg_pool2d(predicted_observed_images, ksize = self.super_resolution_factor, strides = self.super_resolution_factor, padding = 'SAME')

        if self.model_noise:
            predicted_observed_images = predicted_observed_images * self.noise_image_scale + tf.random_normal(shape = self.noise_scale.shape) * self.noise_scale + self.noise_bias

        if self.model_bayer:
            predicted_observed_images = apply_bayer_filter(predicted_observed_images, self.bayer_filters)

        if self.model_demosaic:
            predicted_observed_images = apply_demosaic_filter(predicted_observed_images, self.demosaic_filters)

        return predicted_observed_images


    def call(self, observed_images):
        
        observed_images = self.apply_adc_function(observed_images)
            
        predicted_observed_images = self.predict_observed_images()
        
        # saturating to 1 here intentionally kills gradients in the saturated parts, so we're not penalized
        predicted_observed_images = tf.minimum(1.0, predicted_observed_images)

        self.add_loss(tf.compat.v1.losses.mean_squared_error(observed_images, predicted_observed_images))

    def apply_psf_physicality_constraints(self):
        # apply physicality constraints, PSF must be positive and normal
        self.point_spread_functions.assign(tf.maximum(0, self.point_spread_functions))
        # hmm, why doesn't this help?  maybe it got NaNs when PSFs go to zero with initial training overshoot?
        #self.point_spread_functions.assign(self.point_spread_functions / tf.reduce_sum(self.point_spread_functions, axis = (-4, -3), keepdims = True))

    def apply_image_physicality_constraints(self):
        # apply physicality constraints, image must be positive
        self.estimated_image.assign(tf.maximum(0, self.estimated_image))

        if self.model_adc:
            adc_min = tf.reduce_min(self.adc_function, axis = 0, keepdims = True)
            print("ADC min: {}".format(adc_min.numpy()))
            self.adc_function.assign(self.adc_function - adc_min)
            
            adc_max = tf.reduce_max(self.adc_function, axis = 0, keepdims = True)
            print("ADC max: {}".format(adc_max.numpy()))
            self.adc_function.assign(self.adc_function / adc_max)


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

            print("ADC MSE vs linear: {}".format(tf.compat.v1.losses.mean_squared_error(self.adc_function, self.adc_function_linear).numpy()))
            print("ADC MSE vs   sRGB: {}".format(tf.compat.v1.losses.mean_squared_error(self.adc_function, self.adc_function_srgb).numpy()))

    def write_psf_debug_images(self, output_dir, step):
        write_sequential_image(self.get_psf_examples(), output_dir, "psf_examples", step)

    def write_image_debug_images(self, output_dir, step):        
        write_sequential_image(self.estimated_image, output_dir, "estimated_image", step)
        write_sequential_image(center_image_per_channel(tf.squeeze(self.estimated_image, axis = 0)), output_dir, "estimated_image_centered", step)

        if self.model_noise:
            write_sequential_image(self.noise_bias, output_dir, "noise_bias", step)
            write_sequential_image(self.noise_scale, output_dir, "noise_scale", step)
            write_sequential_image(self.noise_image_scale, output_dir, "noise_image_scale", step)
        
        if self.model_adc:
            self.print_adc_stats()
            write_sequential_image(adc_function_to_graph(self.adc_function), output_dir, "adc", step)

        if self.model_bayer:
            write_sequential_image(self.bayer_filters, output_dir, "bayer", step)
            
        if self.model_demosaic:
            write_sequential_image(demosaic_filters_to_image(self.demosaic_filters), output_dir, "demosaic", step)


    def generate_synthetic_data(truth_image, psf_variance = 4, psf_jitter = 4):
        print("Setting up model for synthetic data...")

        truth_image = tf.image.resize(truth_image, (self.width * self.super_resolution_factor, self.height * self.super_resolution_factor), method = tf.image.ResizeMethod.BICUBIC)
        self.estimated_image.assign(truth_image)

        # rest todo...
        

















