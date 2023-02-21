import tensorflow as tf
import tensorez.modified_stn as stn
import os
import os.path
import gc

from tensorez.util import *
from tensorez.luckiness import *

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
    flow_dataset_filename,
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
    # prevents the result for having too much detail in the flow
    flow_detail_loss_coefficient = 1e-6,
    # prevents the result from encoding in the flow what it could represent in the alignment transform
    flow_alignment_loss_coefficient = 1e-6,
    # number of learning stetps per lod
    max_steps = 500,
    # number of high lods to skip
    skip_lods = 0,
    # we start with low resolution images (LOD = level of detail) and step up to the full resolution image, each step increases resolution by this factor
    lod_factor = 2,
    flow_downsample = 4,
    # the smallest lod is at least this big
    min_size = 32,
    # we blur the target images by this many pixels, to provide slopes to allow the images to roll into place
    blur_pixels = 4,
    # if true, after aligning each image, we add it into our target - could help if each image is bad, but could also add error
    incrementally_improve_target_image = False,
    allow_rotation = True,
    allow_scale = False,
    allow_skew = False,
    normalize_flow = True,
):
    if not allow_rotation:
        learning_rate_rotation = 0
    if not allow_scale:
        learning_rate_log_scale = 0
    if not allow_skew:
        learning_rate_skew = 0

    image_shape = lights[0].shape
    image_shape_hw = image_shape.as_list()[-3:-1]
    flow_shape_hw = [int(image_shape_hw[0] // flow_downsample // 2) * 2, int(image_shape_hw[1] // flow_downsample // 2) * 2]

    flow_dataset = np.lib.format.open_memmap(
        filename=flow_dataset_filename,
        mode='w+',
        dtype=np.float32,
        shape=(len(lights), 2, flow_shape_hw[0], flow_shape_hw[1])
    )

    mask_image = generate_mask_image(image_shape)

    if debug_output_dir is not None:
        write_image(mask_image, os.path.join(debug_output_dir, 'mask_image.png'))

        # eventually we'll generate these outputs in a nicer way, and or not even use them as we'd also want to use lucky imaging
        # so, only doing them when we're doing debug output.
        average_unaligned = tf.Variable(tf.zeros(image_shape))
        average_global_aligned = tf.Variable(tf.zeros(image_shape))
        average_local_aligned = tf.Variable(tf.zeros(image_shape))

    # todo: fft
    if blur_pixels != 0:
        #blur_kernel = gaussian_psf(blur_pixels, standard_deviation = 0.3)
        blur_kernel = tent_psf(blur_pixels)
        # pointless dim for convolution (will be conv2d input_dim)
        blur_kernel = tf.expand_dims(blur_kernel, axis = -2)
        # have to tile, conv2d crashes rather than broadcasting across the last dimension (output_dim)
        blur_kernel = tf.tile(blur_kernel, multiples = (1, 1, 1, 3))

     # could compute different values for width and height, and even for rotation, but KISS:
    max_dimension = max(image_shape_hw[-2], image_shape_hw[-1])
    min_dimension = min(image_shape_hw[-2], image_shape_hw[-1])

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

    average_final_highest_lod_loss = 0
    for image_index, middleward_index in middle_out(target_image_index, len(lights)):
        gc.collect()
        image_bhwc = lights[image_index]

        if debug_output_dir is not None:
            average_unaligned.assign_add(image_bhwc)

        normalized_image_bhwc = image_with_zero_mean_and_unit_variance(image_bhwc)

        prev_flow = None

        # todo: could do some exponential moving average momentum thingy here
        alignment_transforms[image_index, :].assign(alignment_transforms[middleward_index, :])

        for lod in range(num_lods - 1, skip_lods - 1, -1):
            gc.collect()

            lod_downsample_factor = pow(lod_factor, lod)
            lod_image_shape_hw = [int(image_shape_hw[0] // lod_downsample_factor // 2) * 2, int(image_shape_hw[1] // lod_downsample_factor // 2) * 2]
            lod_max_dimension = max(lod_image_shape_hw[-2], lod_image_shape_hw[-1])

            lod_flow_shape_hw = [int(lod_image_shape_hw[0] // flow_downsample // 2) * 2, int(lod_image_shape_hw[1] // flow_downsample // 2) * 2]
            
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
                lod_mask_image = tf.image.resize(mask_image, lod_image_shape_hw, antialias=True)
                lod_target_image = tf.image.resize(target_image, lod_image_shape_hw, antialias=True)
                lod_image = tf.image.resize(normalized_image_bhwc, lod_image_shape_hw, antialias=True)
            else:
                lod_mask_image = mask_image
                lod_target_image = target_image
                lod_image = normalized_image_bhwc

            if blur_pixels != 0:
                lod_target_image = tf.nn.conv2d(lod_target_image, blur_kernel, strides = 1, padding = 'SAME')
                lod_image = tf.nn.conv2d(lod_image, blur_kernel, strides = 1, padding = 'SAME')

            flow = tf.Variable(tf.zeros(shape=(1, 2, lod_flow_shape_hw[-2], lod_flow_shape_hw[-1])))
            # heh, random for testing
            #flow = tf.Variable((tf.random.uniform(shape=(1, 2, lod_image_shape_hw[-2], lod_image_shape_hw[-1])) - .5)*.1)
            
            if prev_flow is not None:
                # annoying permutation, move i to c and back so we can resize, treating flow input coordinate dimension as channels
                upscaled_flow = tf.transpose(prev_flow, perm=(0, 2, 3, 1))
                upscaled_flow = tf.image.resize(upscaled_flow, lod_flow_shape_hw, antialias=True)
                upscaled_flow = tf.transpose(upscaled_flow, perm=(0, 3, 1, 2))
                flow.assign(upscaled_flow)
            prev_flow = flow

            debug_string = f'Aligning image {image_index + 1} of {len(lights)}, lod {num_lods - lod} of {num_lods - skip_lods}'
            print(debug_string)
            
            

            new_alignment_transform, new_flow, final_loss = local_align_training_loop(
                lod_image,
                # it doesn't seem to like using a slice of a variable, so need to make a new one
                alignment_transform=tf.Variable(alignment_transforms[image_index, :]),
                flow=flow,
                target_image=lod_target_image,
                mask_image=lod_mask_image,
                transform_learning_rate=transform_learning_rate_for_lod,
                transform_max_update=transform_max_update_for_lod,
                flow_learning_rate=flow_learning_rate,
                flow_max_update=flow_max_update,
                flow_detail_loss_coefficient=flow_detail_loss_coefficient,
                flow_alignment_loss_coefficient=flow_alignment_loss_coefficient,
                max_steps = tf.constant(max_steps, dtype=tf.int32),
                debug_string=tf.constant(debug_string)

            )

            alignment_transforms[image_index, :].assign(new_alignment_transform)
            flow.assign(new_flow)

        aligned_normalized_image = None
        if incrementally_improve_target_image or (debug_output_dir is not None):
            aligned_normalized_image = transform_image(normalized_image_bhwc, alignment_transforms[image_index, :], flow)

        if incrementally_improve_target_image:
            new_image_weight = 1.0 / (image_index + 1)
            target_image.assign(image_with_zero_mean_and_unit_variance(target_image * (1 - new_image_weight) + normalized_image_bhwc * new_image_weight))

        if debug_output_dir is not None:
            global_aligned = transform_image(image_bhwc, alignment_transforms[image_index, :], flow_bihw = None)
            average_global_aligned.assign_add(global_aligned)

            local_aligned = transform_image(image_bhwc, alignment_transforms[image_index, :], flow_bihw = flow)
            average_local_aligned.assign_add(local_aligned)

        # store it to the disk - magic!
        flow_dataset[image_index:image_index+1, :, :, :] = flow

        average_final_highest_lod_loss += final_loss   

        if debug_output_dir is not None:            
            write_image(local_aligned, os.path.join(debug_output_dir, f"local_aligned_{image_index:08d}.png"))
            if incrementally_improve_target_image:
                write_image(target_image, os.path.join(debug_output_dir, f"target_after_{image_index:08d}.png"))

            write_flow_image(flow, os.path.join(debug_output_dir, f"flow_{image_index:08d}.png"))

    if debug_output_dir is not None:
        # so far we've skipped the target image, add it in
        unnormalized_target_image = lights[target_image_index]
        average_unaligned.assign_add(unnormalized_target_image)
        average_global_aligned.assign_add(unnormalized_target_image)
        average_local_aligned.assign_add(unnormalized_target_image)

        average_unaligned.assign(average_unaligned / len(lights))
        average_global_aligned.assign(average_global_aligned / len(lights))
        average_local_aligned.assign(average_local_aligned / len(lights))

        write_image(average_unaligned, os.path.join(debug_output_dir, "aa_average_unaligned.png"))
        write_image(average_global_aligned, os.path.join(debug_output_dir, "ab_average_global_aligned.png"))
        write_image(average_local_aligned, os.path.join(debug_output_dir, "ad_average_local_aligned.png"))

    if normalize_flow:        
        num_flows = flow_dataset.shape[0]
        average_flow = tf.Variable(tf.zeros(shape=(1, 2, flow_shape_hw[0], flow_shape_hw[1])))
        for image_index in range(num_flows):
            print(f'Averaging flow {image_index+1} of {num_flows}')
            flow = flow_dataset[image_index:image_index+1]
            average_flow.assign_add(flow)
        average_flow.assign(average_flow / num_flows)

        if debug_output_dir is not None:
            write_flow_image(average_flow, os.path.join(debug_output_dir, 'average_flow.png'))

        for image_index in range(num_flows):
            print(f'Normalizing flow {image_index+1} of {num_flows}')

            normalized_flow = flow_dataset[image_index:image_index+1] - average_flow
            if debug_output_dir is not None:
                write_flow_image(normalized_flow, os.path.join(debug_output_dir, f"flow_normalized_{image_index:08d}.png"))

            flow_dataset[image_index:image_index+1] = normalized_flow

        if debug_output_dir is not None:
            average_image = tf.Variable(tf.zeros(image_shape))
            for image_index in range(len(lights)):
                image = lights[image_index]
                image = transform_image(image,  alignment_transforms[image_index, :], flow_dataset[image_index:image_index+1,:])
                average_image.assign_add(image)
            average_image.assign(average_image / len(lights))
            write_image(average_image, os.path.join(debug_output_dir, "ac_average_local_aligned_normalized.png"))

    return alignment_transforms, flow_dataset

def write_flow_image(flow_bihw, filename, flow_scale = 100, style = 'rg', **write_image_kwargs):
    if style == 'rg':
        flow_image = tf.transpose(flow_bihw, perm=(0,2,3,1))
        flow_image = tf.concat([flow_image, tf.zeros(shape = (1,flow_image.shape[1], flow_image.shape[2], 1))], axis=-1)
        flow_image *= flow_scale
        flow_image += 0.5    
        write_image(flow_image, filename, saturate = True, **write_image_kwargs)
    if style == 'hsv_dark' or style == 'hsv_bright':
        flow_bhwi = tf.transpose(flow_bihw, perm=(0,2,3,1))
        flow_mag = tf.sqrt(tf.square(flow_bhwi[..., 0]), tf.square(flow_bhwi[...,1]))
        flow_theta = tf.atan2(flow_bhwi[..., 1], flow_bhwi[..., 0])
        flow_hsv_bhwc = tf.stack(
            [
                tf.math.floormod(flow_theta / ( 2.0 * math.pi), 1.0),            
                tf.ones(shape = (flow_bhwi.shape[0], flow_bhwi.shape[1], flow_bhwi.shape[2])),
                tf.minimum(flow_mag * flow_scale, 1.0)
            ],
            axis=-1
        )
        flow_rgb_bhwc = tf.image.hsv_to_rgb(flow_hsv_bhwc)
        write_image(flow_rgb_bhwc, filename, **write_image_kwargs)


# can't be function because optimizer creates variables - even passing it in didn't help.
#@tf.function#(jit_compile=False)
def local_align_training_loop(image_bhwc, alignment_transform, flow, target_image, mask_image, transform_learning_rate, transform_max_update, flow_learning_rate, flow_max_update, flow_detail_loss_coefficient, flow_alignment_loss_coefficient, max_steps, debug_string):

    # for Adam, need to nerf learning rate a lot or it way overshoots on the first update
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    
    # This one jiggles a lot!
    #optimizer = tf.keras.optimizers.RMSprop()

    # everyone says this is the coolest:
    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001)

    for step in range(max_steps):
        variables = [alignment_transform, flow]

        with tf.GradientTape() as tape:
            tape.watch(variables)
            alignment_guess_image = transform_image(image_bhwc, alignment_transform, flow)
            loss = (
                alignment_loss(alignment_guess_image, target_image, mask_image) +
                flow_detail_loss(flow, flow_detail_loss_coefficient) +
                flow_alignment_loss(flow, flow_alignment_loss_coefficient)
            )

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        #transform_update = transform_gradient * transform_learning_rate
        #transform_update = tf.clip_by_value(transform_update, -transform_max_update, transform_max_update)
        #alignment_transform -= transform_update

        #flow_update = flow_gradient * flow_learning_rate
        #flow_update = tf.clip_by_value(flow_update, -flow_max_update, flow_max_update)
        #flow -= flow_update

        tf.print(debug_string, "loss:", loss, "after step", step, "of", max_steps, output_stream=sys.stdout)

    return alignment_transform, flow, loss

# alignment_transform's last index is [delta_x, delta_y, theta_radians, log_scale_x, log_scale_y, skew_x]
@tf.function(jit_compile=False)
def transform_image(unaligned_image_bhwc, alignment_transform, flow_bihw = None):
    if flow_bihw is not None:
        if flow_bihw.shape[2] != unaligned_image_bhwc.shape[1] or flow_bihw.shape[3] != unaligned_image_bhwc.shape[2]:
            flow_bihw = upscale_flow(flow_bihw, [unaligned_image_bhwc.shape[1], unaligned_image_bhwc.shape[2]])
    return stn.spatial_transformer_network(unaligned_image_bhwc, alignment_transform_to_stn_theta(alignment_transform), flow=flow_bihw)

@tf.function(jit_compile=True)
def alignment_loss(unaligned_image_bhwc, target_image_bhwc, mask_image_bhwc):
    return tf.math.reduce_mean(tf.math.square(tf.math.multiply((unaligned_image_bhwc - target_image_bhwc), mask_image_bhwc)), axis = (-3, -2, -1))

@tf.function(jit_compile=True)
def flow_detail_loss(flow_bihw, flow_detail_loss_coefficient):
    fft_of_flow = spatial_to_frequency_domain(flow_bihw)
    fft_freqs = fft_spatial_frequencies_per_pixel(flow_bihw.shape) * tf.cast(max(flow_bihw.shape[-2], flow_bihw.shape[-1]),dtype=tf.float32)
    
    # mumble mumble Kolmogorov
    fft_freqs = tf.pow(fft_freqs, 5.0 / 3.0)

    return tf.reduce_mean(tf.square(tf.abs(fft_of_flow)) * fft_freqs) * flow_detail_loss_coefficient

@tf.function(jit_compile=True)
def flow_alignment_loss(flow_bihw, flow_alignment_loss_coefficient):
    average_shift_i = tf.reduce_mean(flow_bihw, axis=[0, 2, 3])
    shift_distance_squared = tf.square(average_shift_i[0]) + tf.square(average_shift_i[1])

    return shift_distance_squared * flow_alignment_loss_coefficient


# XLA (jit_compile) breaks with tf.image.resize because the gradient isn't implemented.
@tf.function#(jit_compile=True)
def upscale_flow(flow_bihw, shape_hw):
    flow_bihw = tf.transpose(flow_bihw, perm=(0, 2, 3, 1))
    flow_bihw = tf.image.resize(flow_bihw, tf.stop_gradient(shape_hw), antialias=True)
    flow_bihw = tf.transpose(flow_bihw, perm=(0, 3, 1, 2))
    return flow_bihw
