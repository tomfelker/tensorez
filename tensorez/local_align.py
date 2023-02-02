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
def local_align(
    # an iterable yielding the images to align
    lights,
    # we will write the alignment info in here
    alignment_output_dir,
    # the index of the lights image to align everything else with (default is the middle of the array)
    target_image_index = None,    
    # if not none, we will dump a bunch of debug images here
    debug_output_dir = None,

    learning_rate_flow = 1e-2,
    # how much to update dx and dy per mean-squared-error-of-unit-variance
    learning_rate_translation = 1e-3,
    # how much to update theta by per mean-squared-error-of-unit-variance
    learning_rate_rotation = 1e-3,
    # how much to update log_scale by per mean-squared-error-of-unit-variance
    learning_rate_log_scale = 1e-3,
    # how much to update skew by per mean-squared-error-of-unit-variance
    learning_rate_skew = 1e-3,
    # don't update local flow by more than this many pixels
    max_update_flow_pixels = 0.9,
    # don't update dx or dy by more than this many pixels    
    max_update_translation_pixels = 0.9,
    # don't rotate such that we'd move some part of the image by more than this many pixels
    max_update_rotation_pixels = 3,
    # don't scale such that we'd move some part of the image by more than this many pixels
    max_update_scale_pixels = 3,
    # don't skew such that we'd move some part of the image by more than this many pixels
    max_update_skew_pixels = 3,

    flow_regularization_loss = 0,#1e-8,
    # number of learning stetps per lod
    max_steps = 50,
    # number of high lods to skip
    skip_lods = 0,
    # we start with low resolution images (LOD = level of detail) and step up to the full resolution image, each step increases resolution by this factor
    lod_factor = math.sqrt(2),
    # the smallest lod is at least this big
    min_size = 32,
    # we blur the target images by this many pixels, to provide slopes to allow the images to roll into place
    blur_pixels = 8,
    # if true, after aligning each image, we add it into our target - could help if each image is bad, but could also add error
    incrementally_improve_target_image = False,
    allow_rotation = True,
    allow_scale = False,
    allow_skew = False
):
    if not allow_rotation:
        learning_rate_rotation = 0
    if not allow_scale:
        learning_rate_log_scale = 0
    if not allow_skew:
        learning_rate_skew = 0

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

    # [alignment_param_index: dx, dy, theta, log(sx), log(sy), skew]
    identity_alignment_transform = tf.zeros(6)

    # [batch, alignment_param_index]
    alignment_transforms = tf.Variable(tf.tile(tf.expand_dims(identity_alignment_transform, axis=0), multiples=(len(lights), 1)))

    

    if target_image_index is None:
        target_image_index = int(len(lights) / 2)

    
    target_image = tf.Variable(lights[target_image_index])    
    target_image.assign(image_with_zero_mean_and_unit_variance(target_image))

    
    for image_index, middleward_index in middle_out(target_image_index, len(lights)):
        image_bhwc = lights[image_index]

        image_bhwc = image_with_zero_mean_and_unit_variance(image_bhwc)

        prev_flow = None

        # todo: could do some exponential moving average momentum thingy here
        if image_index > 0:
            alignment_transforms[image_index, :].assign(alignment_transforms[middleward_index, :])

        for lod in range(num_lods - 1, skip_lods - 1, -1):

            lod_downsample_factor = pow(lod_factor, lod)
            lod_hw = [int(image_hw[0] / lod_downsample_factor), int(image_hw[1] / lod_downsample_factor)]
            lod_max_dimension = max(lod_hw[-2], lod_hw[-1])

            transform_learning_rate_for_lod = tf.constant([
                learning_rate_translation * lod_downsample_factor,
                learning_rate_translation * lod_downsample_factor,
                learning_rate_rotation * lod_downsample_factor,
                learning_rate_log_scale * lod_downsample_factor,
                learning_rate_log_scale * lod_downsample_factor,
                learning_rate_skew * lod_downsample_factor,
            ])

            flow_learning_rate = learning_rate_flow * lod_downsample_factor

            transform_max_update_for_lod = tf.constant([
                max_update_translation_pixels / lod_max_dimension,
                max_update_translation_pixels / lod_max_dimension,
                max_update_rotation_pixels / lod_max_dimension,
                math.log(1.0 + max_update_scale_pixels / lod_max_dimension),
                math.log(1.0 + max_update_scale_pixels / lod_max_dimension),
                max_update_skew_pixels / lod_max_dimension,
            ])

            flow_max_update = max_update_flow_pixels / lod_max_dimension

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

            flow = tf.Variable(tf.zeros(shape=(1, 2, lod_hw[-2], lod_hw[-1])))
            #flow = tf.Variable((tf.random.uniform(shape=(1, 2, lod_hw[-2], lod_hw[-1])) - .5)*.1)
            if prev_flow is not None:
                # annoying permutation, move i to c
                upscaled_flow = tf.transpose(prev_flow, perm=(0, 2, 3, 1))
                upscaled_flow = tf.image.resize(upscaled_flow, lod_hw, antialias=True)
                upscaled_flow = tf.transpose(upscaled_flow, perm=(0, 3, 1, 2))
                flow.assign(upscaled_flow)
            prev_flow = flow

            print(f'Aligning image {image_index + 1} of {len(lights)}, lod {num_lods - lod} of {num_lods - skip_lods}')
            
            new_alignment_transform, new_flow = local_align_training_loop(
                lod_image,
                alignment_transform=alignment_transforms[image_index, :],
                flow=flow,
                target_image=lod_target_image,
                mask_image=lod_mask_image,
                transform_learning_rate=transform_learning_rate_for_lod,
                transform_max_update=transform_max_update_for_lod,
                flow_learning_rate=flow_learning_rate,
                flow_max_update=flow_max_update,
                flow_regularization_loss=flow_regularization_loss,
                max_steps = tf.constant(max_steps, dtype=tf.int32)
            )

            alignment_transforms[image_index, :].assign(new_alignment_transform)
            flow.assign(new_flow)
            
                

        aligned_image = None
        if incrementally_improve_target_image or (debug_output_dir is not None):
            aligned_image = transform_image(image_bhwc, alignment_transforms[image_index, :], flow)

        if incrementally_improve_target_image:
            new_image_weight = 1.0 / (image_index + 1)
            target_image.assign(image_with_zero_mean_and_unit_variance(target_image * (1 - new_image_weight) + aligned_image * new_image_weight))

        # todo: save to alignment_output_dir

        if debug_output_dir is not None:            
            write_image(aligned_image, os.path.join(debug_output_dir, f"aligned_{image_index:08d}.png"), normalize = True)
            if incrementally_improve_target_image:
                write_image(target_image, os.path.join(debug_output_dir, f"target_after_{image_index:08d}.png"), normalize = True)

            flow_image = tf.transpose(flow, perm=(0,2,3,1))
            flow_image = tf.concat([flow_image, tf.zeros(shape = (1,flow_image.shape[1], flow_image.shape[2], 1))], axis=-1)
            flow_image = (image_with_zero_mean_and_unit_variance(flow_image) + 1.0) / 2.0
            write_image(flow_image, os.path.join(debug_output_dir, f"flow_{image_index:08d}.png"), saturate=True)

    return alignment_transforms

def read_average_image_with_alignment_transforms(lights, alignment_output_dir, image_shape, dark_image):

    # todo - load from alignment_output_dir
    alignment_transforms = None
    flow = None

    target_image = tf.Variable(tf.zeros(image_shape))
    for image_index, image_bhwc in enumerate(lights):    
        if dark_image is not None:
            image_bhwc = image_bhwc - dark_image

        alignment_transform = alignment_transforms[image_index, :]
        image_bhwc = transform_image(image_bhwc, alignment_transform, flow)
        target_image.assign_add(image_bhwc)

    return target_image / len(lights)


@tf.function
def local_align_training_loop(image_bhwc, alignment_transform, flow, target_image, mask_image, transform_learning_rate, transform_max_update, flow_learning_rate, flow_max_update, flow_regularization_loss, max_steps):

    for step in range(max_steps):
        variable_list = [alignment_transform, flow]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(variable_list)
            alignment_guess_image = transform_image(image_bhwc, alignment_transform, flow)
            guess_loss = alignment_loss(alignment_guess_image, target_image, mask_image) + regularization_loss(flow, flow_regularization_loss)

        transform_gradient = tape.gradient(guess_loss, alignment_transform)
        flow_gradient = tape.gradient(guess_loss, flow)
        del tape

        transform_update = transform_gradient * transform_learning_rate
        transform_update = tf.clip_by_value(transform_update, -transform_max_update, transform_max_update)
        alignment_transform -= transform_update

        flow_update = flow_gradient * flow_learning_rate
        flow_update = tf.clip_by_value(flow_update, -flow_max_update, flow_max_update)
        flow -= flow_update

        tf.print("loss:", guess_loss, "after step", step, "of", max_steps)

    return alignment_transform, flow





# alignment_transform's last index is [delta_x, delta_y, theta_radians, log_scale_x, log_scale_y, skew_x]
@tf.function
def transform_image(unaligned_image_bhwc, alignment_transform, flow):
    
    return stn.spatial_transformer_network(unaligned_image_bhwc, alignment_transform_to_stn_theta(alignment_transform), flow=flow)

@tf.function
def alignment_loss(unaligned_image_bhwc, target_image_bhwc, mask_image_bhwc):
    return tf.math.reduce_mean(tf.math.square(tf.math.multiply((unaligned_image_bhwc - target_image_bhwc), mask_image_bhwc)), axis = (-3, -2, -1))

@tf.function
def regularization_loss(flow, flow_regularization_loss):
    # todo: highpass?
    return tf.reduce_mean(flow) * flow_regularization_loss


