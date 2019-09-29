import numpy as np
import tensorflow as tf
import os
import glob
import datetime

from tensorez.model import *
from tensorez.util import *


# Try to model the ADC curve, in case it's not really sRGB..
# although sadly, sometimes it just learns to kill the bottom of the curve to reduce its error, so it doesn't work great
model_adc = False

# Try to model sensor noise - not really working yet, as the image gets sucked into the noise and causes some weird artifacts (similar to when training too fast?)
model_noise = False

# This actually may be pointless, because the bayer filter only discards information, which the demosaic filter could simply choose to disregard if it wants...
model_bayer = True

# But this is a good idea though
model_demosaic = False

bayer_tile_size = 2

# if the images were already centered, and we're modelling bayer, we'll need to try to realign their bayer filters first... so we wish this were false, but c'est la vie
images_were_centered = False

# if true, we demosaic them for computing averages, but not otherwise
images_are_raw = False

# do the extra step of centering and aligning the images... lets us use smaller PSFs, so should be a win
# but realign currently uses Center of Mass, which only really works for planets, not even ISS... need some autocorrelation based thing...
realign_images = False

# use a larger estimate than the images themselves - should help if the PSFs are fairly small.
super_resolution_factor = 2

bifurcated_training = False


# tunings

# hmm, i have no idea why, but 64 inits way faster than 32
psf_size = 32 * super_resolution_factor

image_count_limit = 15 # 200 oom

psf_training_steps = 1
psf_learning_rate = .001

image_training_steps = 1
image_learning_rate = .001

overall_training_steps = 1000

crop = None

#data selection

#file_glob = os.path.join('data', 'saturn_bright_mvi_6902', '????????.png')
#file_glob = os.path.join('data', 'jupiter_mvi_6906', '????????.png')
#file_glob = os.path.join('obd', 'data','epsilon_lyrae', '????????.png')
#file_glob = os.path.join('data', 'ISS_aligned_from_The_8_Bit_Zombie', '*.tif')
file_glob = os.path.join('data', 'powerline_t4i_raw', '*.cr2'); crop = (512, 512); images_are_raw = True

output_dir = os.path.join("output", "latest_" + datetime.datetime.now().replace(microsecond = 0).isoformat().replace(':', '_'))

tf.enable_eager_execution()

# for now, just a single batch
print("Reading input images...")
num_images = 0

color_images = []
raw_images = []

for filename in glob.glob(file_glob):
    color_image = read_image(filename, to_float = not model_adc, crop = crop, demosaic = True)
    color_images.append(color_image)

    if images_are_raw:
        raw_image = read_image(filename, to_float = not model_adc, crop = crop, demosaic = False)
        raw_images.append(raw_image)
    
    num_images += 1
    if num_images == image_count_limit:
        break
color_images = tf.concat(color_images, axis = -4)
if images_are_raw:
    raw_images = tf.concat(raw_images, axis = -4)
else:
    raw_images = color_images


def reduce_image_to_tile(image, tile_size):
    slices = tf.split(image, num_or_size_splits = image.shape[-3] // tile_size, axis = -3)
    image = tf.reduce_mean(tf.stack(slices, axis = 0), axis = 0)
    slices = tf.split(image, num_or_size_splits = image.shape[-2] // tile_size, axis = -2)
    image = tf.reduce_mean(tf.stack(slices, axis = 0), axis = 0)
    return image    

# todo: must rejigger this to handle shifting both sets of images equally...
if realign_images:
    print("Centering images...")
    # if images were already centered, nothing to lose by shifting smaller than bayer pattern
    # todo: support bayer pattern size other than 2, "even"
    images = center_images(images, only_even_shifts = model_demosaic and not images_were_centered)

    if images_were_centered:
        # we need to re-align them to the bayer tile size

        max_align_steps = 5

        steps_with_no_shifts = 0
        for align_step in range(0, max_align_steps):
            print("Bayer aligning images, step {}".format(align_step))
        
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
model = TensoRezModel(
    psf_size = psf_size,
    model_adc = model_adc,
    model_noise = model_noise,
    model_bayer = model_bayer,
    model_demosaic = model_demosaic,
    bayer_tile_size = bayer_tile_size,
    super_resolution_factor = super_resolution_factor,
    images_were_centered = realign_images)

if model.model_adc:
    write_image(adc_function_to_graph(model.adc_function), os.path.join(output_dir, "initial_adc.png"))

model.compute_initial_estimate(color_images)

model.write_psf_debug_images(output_dir, 0)
model.write_image_debug_images(output_dir, 0)

if bifurcated_training:
    print("Beginning training....")

    image_optimizer = tf.train.AdamOptimizer(image_learning_rate)
    psf_optimizer = tf.train.AdamOptimizer(psf_learning_rate)


    for overall_training_step in range(1, overall_training_steps):

        for psf_training_step in range(0, psf_training_steps):    
            with tf.GradientTape(watch_accessed_variables = False) as psf_tape:
                for var in model.psf_training_vars:
                    psf_tape.watch(var)
                for var in model.losses:
                    psf_tape.watch(var)
                    
                model(raw_images)

            print("Overall step {}, psf step {}: loss {}".format(overall_training_step, psf_training_step, sum(model.losses)))

            grads = psf_tape.gradient(model.losses, model.psf_training_vars)
            psf_optimizer.apply_gradients(zip(grads, model.psf_training_vars))

            model.apply_psf_physicality_constraints()

            #write_image(model.get_psf_examples(), os.path.join(output_dir, "psf_examples_latest_{}.png".format(psf_training_step)))        

        model.write_psf_debug_images(output_dir, overall_training_step)
        
        for image_training_step in range(0, image_training_steps):
            with tf.GradientTape(watch_accessed_variables = False) as image_tape:
                for var in model.image_training_vars:
                    image_tape.watch(var)
                for var in model.losses:
                    image_tape.watch(var)
                model(raw_images)

            grads = image_tape.gradient(model.losses, model.image_training_vars)
            image_optimizer.apply_gradients(zip(grads, model.image_training_vars))

            print("Overall step {}, image step {}: loss {}".format(overall_training_step, image_training_step, sum(model.losses)))

            model.apply_image_physicality_constraints()

        model.write_image_debug_images(output_dir, overall_training_step)

else:
    print("unified training...")

    optimizer = tf.train.AdamOptimizer(image_learning_rate)
    for overall_training_step in range(1, overall_training_steps * (psf_training_steps + image_training_steps)):
        all_vars = model.psf_training_vars + model.image_training_vars
        with tf.GradientTape(watch_accessed_variables = False) as tape:
            for var in all_vars:
                tape.watch(var)            
            for var in model.losses:
                tape.watch(var)
                
            model(raw_images)

        print("Step {} loss {}".format(overall_training_step, sum(model.losses)))

        grads = tape.gradient(model.losses, all_vars)
        optimizer.apply_gradients(zip(grads, all_vars))

        model.apply_psf_physicality_constraints()
        model.apply_image_physicality_constraints()

        if overall_training_step < 10 or overall_training_step % 10 == 0:
            model.write_psf_debug_images(output_dir, overall_training_step)
            model.write_image_debug_images(output_dir, overall_training_step)

        
print("Cool");

    
