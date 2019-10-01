################################################################################
#
# TensoRez
#
# Enhance your images with the magic of reverse-mode automatic differentiation!
#
# 
# There's no GUI or CLI yet, just edit the settings below and run.
#
################################################################################

import numpy as np
import tensorflow as tf
import os
import glob
import datetime
import time
from tensorez.model import *
from tensorez.util import *

################################################################################
#
# Settings:
#


# Estimated image is scaled up by this integer factor in both dimensions.  If
# the input images are fairly sharp, this will help - but takes lots of memory.
#
super_resolution_factor = 2


# Size of the estimated point-spread functions, which blur the estimated image
# in hopefully the same way as the atmosphere and aperture diffraction did.
#
# Should be big enough to comfortably encompass all the blur around a star, but
# of course, bigger makes it run slower.  Since this applies to the estimated
# image, it should be increased in proportion to super_resolution_factor.
#
# Certain sizes, like 64, seem much faster - something about GPU wavefronts?
#
psf_size = 64 * super_resolution_factor



# If true, we will align the images according to their center of mass.
#
# Note!  If you align the images yourself, only shift them by multiples of two
# pixels, or it will be impossible to do Bayer processing.
#
# This works okay for planets, but for other things we'll need a better way.
#
# Aligning is not strictly necessary if psf_size is large enough, as a
# shift in the just becomes a shift in the PSF.
#
align_by_center_of_mass = True


################################################################################
#
# Experimental / broken features


# If true, attempt to model the effect of the camera's bayer filter.  Only
# works if we have raw files that can be loaded without demosaicing them.
#
# Still experimental...
#
attempt_bayer_processing = True


# Try to model the ADC curve, in case it's not really sRGB..
# although sadly, sometimes it just learns to kill the bottom of the curve to reduce its error, so it doesn't work great
model_adc = False

# Try to model sensor noise - not really working yet, as the image gets sucked into the noise and causes some weird artifacts (similar to when training too fast?)
model_noise = False

# When we don't have raw un-demosaiced images, try to figure out what the camera did - doesn't really work...
model_demosaic = False

# Images were externally centered, not in shifts divisible by 2 (bayer tile), so for demosaic to work, try to realign them.
images_were_overcentered = True


# 2 unless you have a really weird camera - but we don't support that yet...
bayer_tile_size = 2


################################################################################
#
# Training settings
#

# Quit after this many steps.
training_steps = 10000

# At each step, move along the gradient by this factor - moving to fast will
# lead to artifacts or instability, and moving too slow will take forever.
learning_rate = .001


# With bifurcated training, we alternate between training the PSFs and the image.
# I don't think this is necessary, and will remove it pending more experiments.
bifurcated_training = False
psf_training_steps = 10
psf_learning_rate = .001
image_training_steps = 10
image_learning_rate = .001


################################################################################
#
# Data selection
#
# Finally, here's where you give your input data:
#


# Only load this many images.  May be necessary to prevent out-of-memory,
# or for testing.  Someday, support for batching may improve this.
#
image_count_limit = None

# Give a (width, height) tuple here to crop the images to a given size around
# their center.  Some data below may reset this.
crop = None

# If cropping, you can specify (x, y) offets relative to the center of the image.
crop_offsets = None

# Uncomment one of these, or add your own:
#

#file_glob = os.path.join('data', 'jupiter_mvi_6906', '????????.png')
#file_glob = os.path.join('data', 'saturn_bright_mvi_6902', '????????.png')
#file_glob = os.path.join('obd', 'data','epsilon_lyrae', '????????.png')
#file_glob = os.path.join('data', 'ISS_aligned_from_The_8_Bit_Zombie', '*.tif'); image_count_limit = 50
file_glob = os.path.join('data', 'powerline_t4i_raw', '*.cr2'); crop = (512, 512); crop_offsets = (0, 1500); align_by_center_of_mass = False
#file_glob = os.path.join('data', 'moon_bottom_mvi_6958', '????????.png'); crop = (512, 512); align_by_center_of_mass = False

# Put the data here (and also some images will in the parent, output/latest)
#
output_dir = os.path.join("output", "latest", datetime.datetime.now().replace(microsecond = 0).isoformat().replace(':', '_'))


#
# End of settings.
#
################################################################################

print("TensoRez v0.1")

tf.compat.v1.enable_eager_execution()

print("Reading input images from {}".format(file_glob))
num_images = 0

images_for_initial_estimate = []
images_for_training = []

for filename in glob.glob(file_glob):
    image = read_image(filename, to_float = not model_adc, crop = crop, crop_offsets = crop_offsets, demosaic = True)

    if align_by_center_of_mass:
        image, shift, shift_axis = center_image(image)
        print("Aligning by center of mass, shifted image by {}".format(shift))
    
    images_for_initial_estimate.append(image)

    if attempt_bayer_processing and image.shape[-1] == 1:
        print("Warning: Not doing Bayer processing because we got a monochrome image.")
        attempt_bayer_processing = False

    if attempt_bayer_processing:
        image = read_image(filename, to_float = not model_adc, crop = crop, crop_offsets = crop_offsets, demosaic = False)
        if image.shape[-1] == 1:
            if align_by_center_of_mass:
                print("Also shifted raw image that amount.")
                image = tf.roll(image, shift = shift, axis = shift_axis)
            
            images_for_training.append(image)
        else:                
            print("Warning: Not doing bayer processing because we couldn't get a non-demosaiced raw file.")
            attempt_bayer_processing = False
    
    num_images += 1
    if num_images == image_count_limit:
        print("Stopping after {} images (image_count_limit).".format(image_count_limit))
        break
    
images_for_initial_estimate = tf.concat(images_for_initial_estimate, axis = -4)
if attempt_bayer_processing:    
    images_for_training = tf.concat(images_for_training, axis = -4)
    print("Will do Bayer processing, have training images of shape {}".format(images_for_training.shape))
else:
    images_for_training = images_for_initial_estimate
    print("Using same images for initial estimate and training, shape {}".format(images_for_training.shape))


def reduce_image_to_tile(image, tile_size):
    slices = tf.split(image, num_or_size_splits = image.shape[-3] // tile_size, axis = -3)
    image = tf.reduce_mean(tf.stack(slices, axis = 0), axis = 0)
    slices = tf.split(image, num_or_size_splits = image.shape[-2] // tile_size, axis = -2)
    image = tf.reduce_mean(tf.stack(slices, axis = 0), axis = 0)
    return image    

# This probably doesn't work...
if model_demosaic and images_were_overcentered:
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
        


model = TensoRezModel(
    psf_size = psf_size,
    model_adc = model_adc,
    model_noise = model_noise,
    model_bayer = attempt_bayer_processing or model_demosaic,
    model_demosaic = model_demosaic,
    bayer_tile_size = bayer_tile_size,
    super_resolution_factor = super_resolution_factor,
    realign_center_of_mass = align_by_center_of_mass)

if model.model_adc:
    write_image(adc_function_to_graph(model.adc_function), os.path.join(output_dir, "initial_adc.png"))

model.compute_initial_estimate(images_for_initial_estimate)

model.write_psf_debug_images(output_dir, 0)
model.write_image_debug_images(output_dir, 0)

if bifurcated_training:
    print("Beginning bifurcated training....")

    image_optimizer = tf.compat.v1.train.AdamOptimizer(image_learning_rate)
    psf_optimizer = tf.compat.v1.train.AdamOptimizer(psf_learning_rate)

    for overall_training_step in range(1, training_steps):

        for psf_training_step in range(0, psf_training_steps):    
            with tf.GradientTape(watch_accessed_variables = False) as psf_tape:
                for var in model.psf_training_vars:
                    psf_tape.watch(var)
                for var in model.losses:
                    psf_tape.watch(var)
                    
                model(images_for_training)

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
                model(images_for_training)

            grads = image_tape.gradient(model.losses, model.image_training_vars)
            image_optimizer.apply_gradients(zip(grads, model.image_training_vars))

            print("Overall step {}, image step {}: loss {}".format(overall_training_step, image_training_step, sum(model.losses)))

            model.apply_image_physicality_constraints()

        model.write_image_debug_images(output_dir, overall_training_step)

else:
    print("Beginning training...")

    all_vars = model.psf_training_vars + model.image_training_vars
    print("Will train variables: {}".format(list(map((lambda var: var.name), all_vars))))

    print("Learning rate: {}".format(learning_rate))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)    
    for overall_training_step in range(1, training_steps + 1):
        
        step_start_time = time.perf_counter()
        print("Step {} / {},".format(overall_training_step, training_steps), end = '', flush = True)
        
        with tf.GradientTape(watch_accessed_variables = False) as tape:
            for var in all_vars:
                tape.watch(var)            
            for var in model.losses:
                tape.watch(var)
                
            model(images_for_training)

        print(" loss {:.5e}".format(sum(model.losses)), end = '', flush = True)

        grads = tape.gradient(model.losses, all_vars)
        optimizer.apply_gradients(zip(grads, all_vars))

        model.apply_psf_physicality_constraints()
        model.apply_image_physicality_constraints()

        step_elapsed_seconds = time.perf_counter() - step_start_time
        steps_left = training_steps - overall_training_step
        seconds_left = steps_left * step_elapsed_seconds       
        print(", step took {:.2f} seconds, done in {}".format(step_elapsed_seconds, datetime.timedelta(seconds = round(seconds_left))))

        if overall_training_step < 10 or overall_training_step % 10 == 0:
            model.write_psf_debug_images(output_dir, overall_training_step)
            model.write_image_debug_images(output_dir, overall_training_step)

        
print("May your skies be clear!");

    
