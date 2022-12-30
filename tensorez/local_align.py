import tensorflow as tf
import tensorez.modified_stn as stn
import os
import os.path

from tensorez.util import *

# Goal of all this is something like:
#
# - Load a sequence of raw images, like perhaps a landscape shot with stars in it, or something with multiple planets, or maybe just one...
# - Optionally, load a mask image, with white showing which parts of the raw image to try and align
# - Optionally, specify a smaller rectangular slice of the image to focus on
# - Do the alignment
# - Receive some data, that we can remember along with the original image name
# - Be able to load the image and do that align thingy

# Tries to align a sequence of similar images.
# Returns a tensor with indexes [batch, alignment_transform_param_index], where
# batch is the index of a corresponding image in @param lights, and
# alignment_transform_index refers to (dx, yd, theta) in pixels and radians.
def compute_alignment_transforms(
    # an iterable yielding the images to align
    lights,
    # will be subtracted from each image before alignment
    dark_image = None,
    # the index of the lights image to align everything else with (default is the middle of the array)
    target_image_index = None,
    # if not none, we will dump a bunch of debug images here
    debug_output_dir = None,
    # how much to update dx and dy per mean-squared-error-of-unit-variance
    learning_rate_translation = 1e-3,
    # how much to update theta by per mean-squared-error-of-unit-variance
    learning_rate_rotation = 1e-3,
    # don't update dx or dy by more than this many pixels
    max_update_translation_pixels = 0.9,
    # don't rotate such that we'd move some part of the image by more than this many pixels
    max_update_rotation_pixels = 3,
    # number of learning stetps per lod
    max_steps = 50,
    # number of high lods to skip
    skip_lods = 0,
    # we start with low resolution images (LOD = level of detail) and step up to the full resolution image, each step increases resolution by this factor
    lod_factor = math.sqrt(2),
    # the smallest lod is at least this big
    min_size = 32,
    # we blur the target images by this many pixels, to provide slopes to allow the images to roll into place
    blur_pixels = 16,
    # if true, after aligning each image, we add it into our target - could help if each image is bad, but could also add error
    incrementally_improve_target_image = False,
    # if true, normalize all the alignments - otherwise target_image_index would have an alignent of 0 and the others will be aligned as necessary.
    normalize_alignments = False,
):
    image_shape = lights[0].shape
    image_hw = image_shape.as_list()[-3:-1]

    mask_image = generate_mask_image(image_shape)

    if debug_output_dir is not None:
        write_image(mask_image, os.path.join(debug_output_dir, 'mask_image.png'))

    # todo: fft
    if blur_pixels != 0:
        #blur_kernel = gaussian_psf(blur_pixels, standard_deviation = 0.3)
        blur_kernel = tent_psf(blur_pixels)
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
    target_image.assign(image_with_zero_mean_and_unit_variance(target_image))

    for image_index, middleward_index in middle_out(target_image_index, len(lights)):
        image_bhwc = lights[image_index]
        if dark_image is not None:
            image_bhwc = image_bhwc - dark_image

        image_bhwc = image_with_zero_mean_and_unit_variance(image_bhwc)

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
            aligned_image = transform_image(image_bhwc, alignment_transforms[image_index, :])

        if incrementally_improve_target_image:
            new_image_weight = 1.0 / (image_index + 1)
            target_image.assign(image_with_zero_mean_and_unit_variance(target_image * (1 - new_image_weight) + aligned_image * new_image_weight))

        if debug_output_dir is not None:            
            write_image(aligned_image, os.path.join(debug_output_dir, f"aligned_{image_index:08d}.png"), normalize = True)
            if incrementally_improve_target_image:
                write_image(target_image, os.path.join(debug_output_dir, f"target_after_{image_index:08d}.png"), normalize = True)

    if normalize_alignments:
        # todo: hmm, this is only correct if the rotation happens after te 
        alignment_transforms.assign(alignment_transforms - tf.reduce_mean(alignment_transforms, axis = 0, keepdims=True))

    return alignment_transforms

def read_average_image_with_alignment_transforms(lights, alignment_transforms, image_shape, dark_image):

    target_image = tf.Variable(tf.zeros(image_shape))
    for image_index, image_bhwc in enumerate(lights):    
        if dark_image is not None:
            image_bhwc = image_bhwc - dark_image

        alignment_transform = alignment_transforms[image_index, :]
        image_bhwc = transform_image(image_bhwc, alignment_transform)
        target_image.assign_add(image_bhwc)

    return target_image / len(lights)


@tf.function
def align_image_training_loop(image_bhwc, alignment_transform, target_image, mask_image, learning_rate_for_lod, max_update_for_lod, max_steps):

    for step in range(max_steps):
        variable_list = [alignment_transform]

        with tf.GradientTape() as tape:
            tape.watch(variable_list)
            alignment_guess_image = transform_image(image_bhwc, alignment_transform)
            guess_loss = alignment_loss(alignment_guess_image, target_image, mask_image)

        gradient = tape.gradient(guess_loss, alignment_transform)
        transform_update = gradient * learning_rate_for_lod
        transform_update = tf.clip_by_value(transform_update, -max_update_for_lod, max_update_for_lod)
        alignment_transform -= transform_update
        tf.print("loss:", guess_loss, "after step", step, "of", max_steps)

    return alignment_transform


def middle_out(start_index, count):
    for index in range(start_index + 1, count):
        yield index, index - 1
    for index in range(start_index - 1, -1, -1):
        yield index, index + 1


def generate_mask_image(shape_bhwc, **kwargs):
    mask_image_w = generate_mask_image_1d(shape_bhwc[-2], **kwargs)
    mask_image_h = generate_mask_image_1d(shape_bhwc[-3], **kwargs)

    mask_image_b_wc = tf.reshape(mask_image_w, (1, 1, shape_bhwc[-2], 1))
    mask_image_bh_c = tf.reshape(mask_image_h, (1, shape_bhwc[-3], 1, 1))

    mask_image_bhwc = tf.multiply(mask_image_b_wc, mask_image_bh_c)

    return mask_image_bhwc


def generate_mask_image_1d(size, border_fraction = .1, ramp_fraction = .1, dtype = tf.float32):

    mask_image_1d = tf.linspace(0.0, 1.0 / ramp_fraction, size)
    mask_image_1d = tf.minimum(mask_image_1d, 1.0 / ramp_fraction - mask_image_1d)
    mask_image_1d = tf.clip_by_value(mask_image_1d - border_fraction / ramp_fraction, 0, 1)
    mask_image_1d = tf.cast(mask_image_1d, dtype)

    return mask_image_1d


@tf.function
def transform_image(unaligned_image_bhwc, alignment_transform):

    dx = alignment_transform[0]
    dy = alignment_transform[1]
    theta = alignment_transform[2]

    stn_transform = tf.stack([
        tf.cos(theta), -tf.sin(theta), dx,
        tf.sin(theta), tf.cos(theta), dy])
    stn_transform = tf.reshape(stn_transform, (2, 3))

    return stn.spatial_transformer_network(unaligned_image_bhwc, stn_transform)

@tf.function
def alignment_loss(unaligned_image_bhwc, target_image_bhwc, mask_image_bhwc):
    return tf.math.reduce_mean(tf.math.square(tf.math.multiply((unaligned_image_bhwc - target_image_bhwc), mask_image_bhwc)), axis = (-3, -2, -1))
     

