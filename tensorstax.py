import numpy as np
import tensorflow as tf
import os
import glob
from tensorstax_util import *
from tensorstax_model import *

# Try to model the ADC curve, in case it's not really sRGB..
# although sadly, sometimes it just learns to kill the bottom of the curve to reduce its error, so it doesn't work great
model_adc = False

# Try to model sensor noise - not really working yet, as the image gets sucked into the noise and causes some weird artifacts (similar to when training too fast?)
model_noise = False

# makes sense for now, but won't really work with noise or debayering...
center_images_beforehand = True

super_resolution_factor = 4

psf_size = 64
image_count_limit = 50

psf_training_steps = 50
psf_learning_rate = .001

image_training_steps = 50
image_learning_rate = .001

overall_training_steps = 100

#file_glob = os.path.join('data', 'saturn_bright_mvi_6902', '????????.png')
#file_glob = os.path.join('data', 'jupiter_mvi_6906', '????????.png')
#file_glob = os.path.join('obd', 'data','epsilon_lyrae', '????????.png')

file_glob = os.path.join('data', 'ISS_aligned_from_The_8_Bit_Zombie', '*.tif')



output_dir = os.path.join("output", "latest")


tf.enable_eager_execution()

# for now, just a single batch
print("Reading input images...")
num_images = 0
images = []
for filename in glob.glob(file_glob):
    images.append(read_image(filename, to_float = not model_adc))
    num_images += 1
    if num_images == image_count_limit:
        break
images = tf.stack(images, axis = 0)

if center_images_beforehand:
    print("Centering images...")
    images = center_images(images, only_even_shifts = True)


print("Instantiating model.")
model = TensorstaxModel(psf_size = psf_size, model_adc = model_adc, model_noise = model_noise, super_resolution_factor = super_resolution_factor, images_were_centered = center_images_beforehand)

if model.model_adc:
    write_image(adc_function_to_graph(model.adc_function), os.path.join(output_dir, "initial_adc.png"))

print("Beginning training....")

image_optimizer = tf.train.AdamOptimizer(image_learning_rate)

psf_optimizer = tf.train.AdamOptimizer(psf_learning_rate)

for overall_training_step in range(0, overall_training_steps):

    # init these inside the loop, so their momenta don't persist across updates
    


    for psf_training_step in range(0, psf_training_steps):    
        with tf.GradientTape() as psf_tape:
            model(images)

        print("Overall step {}, psf step {}: loss {}".format(overall_training_step, psf_training_step, sum(model.losses)))

        grads = psf_tape.gradient(model.losses, model.psf_training_vars)
        psf_optimizer.apply_gradients(zip(grads, model.psf_training_vars))

        model.apply_psf_physicality_constraints()

        #write_image(model.get_psf_examples(), os.path.join(output_dir, "psf_examples_latest_{}.png".format(psf_training_step)))        

    write_sequential_image(model.get_psf_examples(), os.path.join(output_dir, "psf_examples"), overall_training_step)    

    for image_training_step in range(0, image_training_steps):
        with tf.GradientTape() as image_tape:
            model(images)

        grads = image_tape.gradient(model.losses, model.image_training_vars)
        image_optimizer.apply_gradients(zip(grads, model.image_training_vars))

        print("Overall step {}, image step {}: loss {}".format(overall_training_step, image_training_step, sum(model.losses)))

        model.apply_image_physicality_constraints()

    estimated_image = model.estimated_image
    write_sequential_image(model.estimated_image, os.path.join(output_dir, "estimated_image"), overall_training_step)
    write_sequential_image(center_image_per_channel(estimated_image), os.path.join(output_dir, "estimated_image_centered"), overall_training_step)

    if model.model_noise:
        write_sequential_image(model.noise_bias, os.path.join(output_dir, "noise_bias"), overall_training_step)
        write_sequential_image(model.noise_scale, os.path.join(output_dir, "noise_scale"), overall_training_step)
        write_sequential_image(model.noise_image_scale, os.path.join(output_dir, "noise_image_scale"), overall_training_step)
    
    if model.model_adc:
        model.print_adc_stats()
        write_sequential_image(adc_function_to_graph(model.adc_function), os.path.join(output_dir, "adc"), overall_training_step)
    

print("Cool");

    
