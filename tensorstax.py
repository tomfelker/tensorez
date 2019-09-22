import numpy as np
import tensorflow as tf
import os
import glob
from tensorstax_util import *
from tensorstax_model import *


psf_size = 128
batch_size = 40

psf_training_steps = 10
psf_learning_rate = .001

image_training_steps = 10
image_learning_rate = .001

overall_training_steps = 100

data_path = os.path.join('data', 'saturn_bright_mvi_6902')
#data_path = os.path.join('data', 'jupiter_mvi_6906')
#data_path = os.path.join('obd', 'data','epsilon_lyrae')

file_glob = os.path.join(data_path, '????????.png')

output_dir = "output"


tf.enable_eager_execution()

# for now, just a single batch
print("Reading input images...")
num_images = 0
images = []
for filename in glob.glob(file_glob):
    images.append(read_image(filename))
    num_images += 1
    if num_images == batch_size:
        break
images = tf.stack(images, axis = 0)

print("Instantiating model.")
model = TensorstaxModel(psf_size = psf_size)

print("Beginning training....")

image_optimizer = tf.train.AdamOptimizer(image_learning_rate)

psf_optimizer = tf.train.AdamOptimizer(psf_learning_rate)

for overall_training_step in range(0, overall_training_steps):

    # init these inside the loop, so their momenta don't persist across updates
    


    for psf_training_step in range(0, psf_training_steps):    
        with tf.GradientTape() as psf_tape:
            model(images)

        print("Overall step {}, psf step {}: loss {}".format(overall_training_step, psf_training_step, sum(model.losses)))

        psf_training_vars = [model.point_spread_functions]
        grads = psf_tape.gradient(model.losses, psf_training_vars)
        psf_optimizer.apply_gradients(zip(grads, psf_training_vars))

        # apply physicality constraints, PSF must be positive and normal
        tf.assign(model.point_spread_functions, tf.maximum(0, model.point_spread_functions))
        # hmm, why doesn't this help?
        #tf.assign(model.point_spread_functions, model.point_spread_functions / tf.reduce_sum(model.point_spread_functions, axis = (-4, -3), keepdims = True))

    for image_training_step in range(0, image_training_steps):
        with tf.GradientTape() as image_tape:
            model(images)

        image_training_vars = [model.estimated_image]
        grads = image_tape.gradient(model.losses, image_training_vars)
        image_optimizer.apply_gradients(zip(grads, image_training_vars))

        print("Overall step {}, image step {}: loss {}".format(overall_training_step, image_training_step, sum(model.losses)))

        # apply physicality constraints, image must be positive
        tf.assign(model.estimated_image, tf.maximum(0, model.estimated_image))

    write_image(model.estimated_image, os.path.join(output_dir, "estimated_image_latest.png"))
    psf_example = model.point_spread_functions[0, :, :, 0, :]
    psf_example = psf_example / tf.reduce_max(psf_example)
    write_image(psf_example, os.path.join(output_dir, "psf_0_latest.png"))
    write_image(model.estimated_image, os.path.join(output_dir, "estimated_image_{}.png".format(overall_training_step)))

print("Cool");

    
