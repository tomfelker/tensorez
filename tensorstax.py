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

model_bayer = True
bayer_tile_size = 2

# if the images were already centered, and we're modelling bayer, we'll need to try to realign their bayer filters first... so we wish this were false, but c'est la vie
images_were_centered = True

# do the extra step of centering and aligning the images... lets us use smaller PSFs, so should be a win
realign_images = True

# use a larger estimate than the images themselves - should help if the PSFs are fairly small.
super_resolution_factor = 2

# tunings
psf_size = 32
image_count_limit = 5

psf_training_steps = 10
psf_learning_rate = .001

image_training_steps = 10
image_learning_rate = .001

overall_training_steps = 100

#data selection

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

def reduce_image_to_tile(image, tile_size):
    slices = tf.split(image, num_or_size_splits = image.shape[-3] // tile_size, axis = -3)
    image = tf.reduce_mean(tf.stack(slices, axis = 0), axis = 0)
    slices = tf.split(image, num_or_size_splits = image.shape[-2] // tile_size, axis = -2)
    image = tf.reduce_mean(tf.stack(slices, axis = 0), axis = 0)
    return image    

if realign_images:
    print("Centering images...")
    # if images were already centered, nothing to lose by shifting smaller than bayer pattern
    # todo: support bayer pattern size other than 2, "even"
    images = center_images(images, only_even_shifts = not images_were_centered)

    if images_were_centered:
        # we need to re-align them to the bayer tile size

        max_align_steps = 5

        steps_with_no_shifts = 0
        for align_step in range(0, max_align_steps):
            print("Aligning images, step {}".format(align_step))
        
            #average_image = tf.reduce_sum(images, axis = 0)
            average_image = images[0]
            average_tile = reduce_image_to_tile(average_image, tile_size = bayer_tile_size)

            write_image(average_tile, os.path.join(output_dir, 'aligned_images', 'average_tile_step{:08}.png'.format(align_step)), normalize = True)

            aligned_images = []
            any_nonzero_shifts = False
            for image in tf.unstack(images, axis = 0):
                best_alignment = None
                best_x_shift = None
                best_y_shift = None

                image_tile = reduce_image_to_tile(image, tile_size = bayer_tile_size)

                for y_shift in range(0, bayer_tile_size):
                    for x_shift in range(0, bayer_tile_size):
                        shifted_tile = tf.roll(image_tile, shift = (y_shift, x_shift), axis = (-3, -2))
                        alignment = tf.compat.v1.losses.mean_squared_error(average_tile, shifted_tile)
                        if best_alignment is None or alignment < best_alignment:
                            best_alignment = alignment
                            best_x_shift = x_shift
                            best_y_shift = y_shift

                if best_x_shift is not 0 and best_y_shift is not 0:
                    any_nonzero_shifts = True                
                aligned_images.append(tf.roll(image, shift = (best_y_shift, best_x_shift), axis = (-3, -2)))
            images = tf.stack(aligned_images, axis = 0)
            if not any_nonzero_shifts:
                steps_with_no_shifts += 1
                if steps_with_no_shifts >= 2:
                    print("Alignment converged!")
                    break
        if any_nonzero_shifts:
            print("Warning: image bayer alignment didn't converge")
        write_aligned_images = True
        if write_aligned_images:
            for image_index, image in enumerate(tf.unstack(images, axis = 0)):
                write_image(image, os.path.join(output_dir, 'aligned_images', 'image_{:08}.png'.format(image_index)))
            


print("Instantiating model.")
model = TensorstaxModel(
    psf_size = psf_size,
    model_adc = model_adc,
    model_noise = model_noise,
    model_bayer = model_bayer,
    bayer_tile_size = bayer_tile_size,
    super_resolution_factor = super_resolution_factor,
    images_were_centered = realign_images)

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

    
