import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

import os
import os.path
import tensorez.align as align
import tensorez.util as util
from tensorez.image_sequence import *
import gc
import glob


output_path = os.path.join('output', 'align_test')

lights = ImageSequence(os.path.join('data', '2022-01-11_jwst', '2022-01-11-0941_0-CapObj.SER'), frame_step = 1)
darks = ImageSequence(os.path.join('data', '2022-01-11_jwst', '2022-01-11-1027_1-CapObj.SER'), frame_step = 1)

#file_glob = os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'lights', '*.SER'); frame_step = 10
#file_glob_darks = None # os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'darks', '*.SER');

learning_rate_translation = 1e-3
learning_rate_rotation = learning_rate_translation
max_update_translation_pixels = 0.9
max_update_rotation_pixels = 3 #0.9
max_steps = 50
skip_lods = 0
lod_factor = 1.3
min_size = 32
blur_pixels = 16
incrementally_improve_target_image = False

def read_average_image_with_alignment_transforms(lights, alignment_transforms, image_shape, dark_image):

    target_image = tf.Variable(tf.zeros(image_shape))
    for image_index, image_bhwc in enumerate(lights):    
        if dark_image is not None:
            image_bhwc = image_bhwc - dark_image

        alignment_transform = alignment_transforms[image_index, :]
        image_bhwc = align.transform_image(image_bhwc, alignment_transform)
        target_image.assign_add(image_bhwc)

    return target_image / len(lights)

@tf.function
def with_zero_mean_and_unit_variance(image_bhwc):
    mean, variance = tf.nn.moments(image_bhwc, axes = (-3, -2))
    stdev = tf.sqrt(variance)
    return (image_bhwc - mean) / stdev


@tf.function
def align_image_training_loop(image_bhwc, alignment_transform, target_image, mask_image, learning_rate_for_lod, max_update_for_lod, max_steps):

    for step in range(max_steps):
        variable_list = [alignment_transform]

        with tf.GradientTape() as tape:
            tape.watch(variable_list)
            alignment_guess_image = align.transform_image(image_bhwc, alignment_transform)
            guess_loss = align.alignment_loss(alignment_guess_image, target_image, mask_image)

        gradient = tape.gradient(guess_loss, alignment_transform)
        transform_update = gradient * learning_rate_for_lod
        transform_update = tf.clip_by_value(transform_update, -max_update_for_lod, max_update_for_lod)
        alignment_transform -= transform_update
        tf.print("loss: ", guess_loss, " after step ", step, " of ", max_steps)

    return alignment_transform


def middle_out(start_index, count):
    for index in range(start_index + 1, count):
        yield index, index - 1
    for index in range(start_index - 1, -1, -1):
        yield index, index + 1

def compute_alignment_transforms(lights, dark_image = None, target_image_index = None, debug_output_dir = None):
    image_shape = lights[0].shape
    image_hw = image_shape.as_list()[-3:-1]

    mask_image = align.generate_mask_image(image_shape)
    util.write_image(mask_image, os.path.join(output_path, 'mask_image.png'))

    # todo: fft
    if blur_pixels != 0:
        #blur_kernel = gaussian_psf(blur_pixels, standard_deviation = 0.3)
        blur_kernel = util.tent_psf(blur_pixels)
        # pointless dim for convolution (will be conv2d input_dim)
        blur_kernel = tf.expand_dims(blur_kernel, axis = -2)
        # have to tile, conv2d crashes rather than broadcasting across the last dimension (output_dim)
        blur_kernel = tf.tile(blur_kernel, multiples = (1, 1, 1, 3))

     # could compute different values for width and height, and even for rotation, but KISS:
    max_dimension = max(image_hw[-2], image_hw[-1])
    min_dimension = min(image_hw[-2], image_hw[-1])

    num_lods = 0
    while pow(lod_factor, -num_lods) * min_dimension > min_size:
        num_lods += 1
    print(f"num_lods: {num_lods}")

    # [alignment_param_index: dx, dy, theta]
    identity_alignment_transform = tf.zeros(3)

    # [batch, alignment_param_index]
    alignment_transforms = tf.Variable(tf.tile(tf.expand_dims(identity_alignment_transform, axis=0), multiples=(len(lights), 1)))


    if target_image_index is None:
        target_image_index = int(len(lights) / 2)

    
    target_image = tf.Variable(lights[target_image_index])
    if dark_image is not None:
        target_image.assign_sub(dark_image)
    target_image.assign(with_zero_mean_and_unit_variance(target_image))

    # todo: middle-out
    for image_index, middleward_index in middle_out(target_image_index, len(lights)):
        image_bhwc = lights[image_index]
        if dark_image is not None:
            image_bhwc = image_bhwc - dark_image

        image_bhwc = with_zero_mean_and_unit_variance(image_bhwc)

        # todo: could do some exponential moving average momentum thingy here
        if image_index > 0:
            alignment_transforms[image_index, :].assign(alignment_transforms[middleward_index, :])

        for lod in range(num_lods - 1, skip_lods - 1, -1):

            lod_downsample_factor = pow(lod_factor, lod)
            lod_hw = [int(image_hw[0] / lod_downsample_factor), int(image_hw[1] / lod_downsample_factor)]
            lod_max_dimension = max(lod_hw[-2], lod_hw[-1])

            learning_rate_for_lod = tf.constant([
                learning_rate_translation * lod_downsample_factor,
                learning_rate_translation * lod_downsample_factor,
                learning_rate_rotation * lod_downsample_factor
            ])

            max_update_for_lod = tf.constant([
                max_update_translation_pixels / lod_max_dimension,
                max_update_translation_pixels / lod_max_dimension,
                max_update_rotation_pixels / lod_max_dimension,
            ])

            max_gradient_for_lod = max_update_for_lod / learning_rate_for_lod
            
            # todo: memoize downsampled_mask_image across lods
            # todo: if we don't incrementally_improve_target_image, we could memoize it too
            if lod_downsample_factor != 1:
                lod_mask_image = tf.image.resize(mask_image, lod_hw, antialias=True)
                lod_target_image = tf.image.resize(target_image, lod_hw, antialias=True)
                lod_image = tf.image.resize(image_bhwc, lod_hw, antialias=True)
            else:
                lod_mask_image = mask_image
                lod_target_image = target_image
                lod_image = image_bhwc

            if blur_pixels != 0:
                lod_target_image = tf.nn.conv2d(lod_target_image, blur_kernel, strides = 1, padding = 'SAME')
                lod_image = tf.nn.conv2d(lod_image, blur_kernel, strides = 1, padding = 'SAME')

            print(f'Aligning image {image_index} of {len(lights)}, lod {num_lods - lod} of {num_lods - skip_lods}')
            
            alignment_transforms[image_index, :].assign(
                align_image_training_loop(
                    lod_image,
                    alignment_transform=alignment_transforms[image_index, :],
                    target_image=lod_target_image,
                    mask_image=lod_mask_image,
                    learning_rate_for_lod=learning_rate_for_lod,
                    max_update_for_lod=max_update_for_lod,
                    max_steps = tf.constant(max_steps, dtype=tf.int32)
                )
            )
                

        aligned_image = None
        if incrementally_improve_target_image or (debug_output_dir is not None):
            aligned_image = align.transform_image(image_bhwc, alignment_transforms[image_index, :])

        if incrementally_improve_target_image:
            new_image_weight = 1.0 / (image_index + 1)
            target_image.assign(with_zero_mean_and_unit_variance(target_image * (1 - new_image_weight) + aligned_image * new_image_weight))

        if debug_output_dir is not None:            
            util.write_image(aligned_image, os.path.join(debug_output_dir, f"aligned_{image_index:08d}.png"), normalize = True)
            if incrementally_improve_target_image:
                util.write_image(target_image, os.path.join(debug_output_dir, f"target_after_{image_index:08d}.png"), normalize = True)

    alignment_transforms.assign(alignment_transforms - tf.reduce_mean(alignment_transforms, axis = 0, keepdims=True))
    return alignment_transforms

dark_image = None
if darks is not None:
    dark_image = darks.read_average_image()

alignment_transforms = compute_alignment_transforms(lights, dark_image=dark_image, debug_output_dir=output_path)

final_average_image_bhwc = read_average_image_with_alignment_transforms(lights, alignment_transforms, lights[0].shape, dark_image)
util.write_image(final_average_image_bhwc, os.path.join(output_path, 'final_average_image.png'))


for image_index, image_bhwc in enumerate(lights):
    if dark_image is not None:
        image_bhwc -= dark_image

    alignment_transform = alignment_transforms[image_index, :]
    transformed_image_bhwc = align.transform_image(image_bhwc, alignment_transform)
    util.write_image(transformed_image_bhwc, os.path.join(output_path, "final_aligned_{:08d}.png".format(image_index)))