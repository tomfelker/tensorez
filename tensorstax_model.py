import numpy as np
import tensorflow as tf

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


    def __init__(self, psf_size = 32):
        super(TensorstaxModel, self).__init__()
        self.psf_size = psf_size


    def build(self, input_shape):
        (self.batch_size, self.width, self.height, self.channels) = input_shape
        print("Building model: batch_size {}, width {}, height {}, channels {}".format(self.batch_size, self.width, self.height, self.channels))

        self.estimated_image = tf.Variable(tf.zeros((self.width, self.height, self.channels)))
        self.have_initial_estimate = False

        self.point_spread_functions = tf.Variable(self.default_point_spread_functions())


    def default_point_spread_functions(self):
        psfs_shape = (self.batch_size, self.psf_size, self.psf_size, 1, self.channels)
        
        #psfs = tf.random.uniform(psfs_shape)
        #psfs = psfs / tf.reduce_sum(psfs, axis = (-4, -3), keepdims = True)

        psfs = tf.zeros(psfs_shape)
        return psfs


    def call(self, observed_images):
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



