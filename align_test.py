import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

import os
import os.path
import tensorez.align as align
import tensorez.util as util
import gc
import glob


output_path = os.path.join('output', 'align_test')

#file_glob = os.path.join('data', '2022-01-11_jwst', '2022-01-11-0941_0-CapObj.SER'); frame_step = 1
#file_glob_darks = os.path.join('data', '2022-01-11_jwst', '2022-01-11-1027_1-CapObj.SER')

file_glob = os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'lights', '*.SER'); frame_step = 10
file_glob_darks = None # os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'darks', '*.SER');

learning_rate_translation = 1e-3
learning_rate_rotation = learning_rate_translation / 3
max_update_translation_pixels = float('inf') # 0.9
max_update_rotation_pixels = float('inf') #0.9
max_steps = 30
passes_per_lod = 30
skip_lods = 3
lod_factor = 1.3
min_size = 128
blur_pixels = 17
weightedness = 50
participation_prize = .01

def read_average_image_with_alignment_transforms(file_glob, alignment_transforms, losses, image_shape, dark_image):
    weights = tf.nn.softmax(weightedness * -losses)
    weights = weights + participation_prize
    weights = weights / tf.reduce_sum(weights)

    #print(f'weights: {weights}')

    target_image = tf.zeros(image_shape)
    image_index = -1
    image_count = 0
    for filename in glob.glob(file_glob):
        for image_bhwc, frame_index in util.ImageSequenceReader(filename, step = frame_step, to_float = True, demosaic = True):
            image_index += 1
            image_count += 1

            # must be in the loop
            if dark_image is not None:
                image_bhwc = image_bhwc - dark_image

            alignment_transform = alignment_transforms[image_index, :]
            image_bhwc = align.transform_image(image_bhwc, alignment_transform)
            target_image = target_image + image_bhwc * weights[image_index]

    return target_image

@tf.function
def with_zero_mean_and_unit_variance(image_bhwc):
    mean, variance = tf.nn.moments(image_bhwc, axes = (-3, -2))
    stdev = tf.sqrt(variance)
    return (image_bhwc - mean) / stdev


@tf.function
def align_image_training_step(image_bhwc, alignment_transform, target_image, mask_image, learning_rate_for_lod, max_update_for_lod):
    variable_list = [alignment_transform]

    with tf.GradientTape() as tape:
        tape.watch(variable_list)
        alignment_guess_image = align.transform_image(image_bhwc, alignment_transform)
        guess_loss = align.alignment_loss(alignment_guess_image, target_image, mask_image)

    gradient = tape.gradient(guess_loss, alignment_transform)
    transform_update = gradient * learning_rate_for_lod
    transform_update = tf.clip_by_value(transform_update, -max_update_for_lod, max_update_for_lod)
    updated_alignment_transform = alignment_transform - transform_update
    return updated_alignment_transform, guess_loss
    



def align_images(file_glob, dark_image):

    # loop below really doesn't need the full average, but we do need the shape and image count, so may as well make a pass
    average_image_bhwc, image_count = util.load_average_image(file_glob, step = frame_step)

    image_shape = average_image_bhwc.shape
    image_hw = image_shape.as_list()[-3:-1]

    if dark_image is not None:
        average_image_bhwc = average_image_bhwc - dark_image
            
    mask_image = align.generate_mask_image(tf.shape(average_image_bhwc))
    util.write_image(mask_image, os.path.join(output_path, 'mask_image.png'))

    util.write_image(average_image_bhwc, os.path.join(output_path, 'initial_average_image.png'), normalize = True)


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


    # dx, dy, theta
    identity_alignment_transform = tf.zeros(3)

    alignment_transforms = tf.Variable(tf.tile(tf.expand_dims(identity_alignment_transform, axis=0), multiples=(image_count, 1)))

    losses = tf.Variable(tf.ones(image_count))

    #todo: really it should be "align this image with the average so far in this pass", rather than the current embarassingly-parallel approach...
    #that has some annoying asymmetry to it (maybe helped with exponential moving average?) if there's individual best frames, but makes the problem easier
    # also todo: instead of this downsampilng business, just do a big fft blur, perhaps with a gaussian or exponential distribution...


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
        
        for pass_index in range(passes_per_lod):
            gc.collect()

            print(f'lod {num_lods - lod} of {num_lods - skip_lods}, pass {pass_index + 1} of {passes_per_lod}')

            #alignment_transforms.assign(alignment_transforms - tf.reduce_mean(alignment_transforms, axis = 0, keepdims=True))

            debug = True
            if debug:
                equal_weight_image = read_average_image_with_alignment_transforms(file_glob, alignment_transforms, tf.ones_like(losses), image_shape, dark_image)
                util.write_image(equal_weight_image, os.path.join(output_path, 'equal_weights_image.png'))
                del equal_weight_image

            # recompute the average image, but with the transforms applied
            target_image = read_average_image_with_alignment_transforms(file_glob, alignment_transforms, losses, image_shape, dark_image)

            # todo: normalize brightness and contrast

            # now we downsample the target (should be equivalent to doing it in the loop above because linearity)
            if lod_downsample_factor != 1:
                target_image = tf.image.resize(target_image, lod_hw, antialias=True)
                downsampled_mask_image = tf.image.resize(mask_image, lod_hw, antialias=True)
            else:
                downsampled_mask_image = mask_image

            # todo: lerp over pass?
            # not blurring at the end explodes it
            if blur_pixels != 0: #and lod_downsample_factor != 1:
                target_image = tf.nn.conv2d(target_image, blur_kernel, strides = 1, padding = 'SAME')

            util.write_image(target_image, os.path.join(output_path, 'target_image.png'))

            target_image = with_zero_mean_and_unit_variance(target_image)

            # now try to align everything to the new target
            image_index = -1
            for filename in glob.glob(file_glob):
                for image_bhwc, frame_index in util.ImageSequenceReader(filename, step = frame_step, to_float = True, demosaic = True):
                    gc.collect()

                    image_index += 1

                    if dark_image is not None:
                        image_bhwc = image_bhwc - dark_image

                    if lod_downsample_factor != 1:
                        image_bhwc = tf.image.resize(image_bhwc, lod_hw, antialias=True)

                    image_bhwc = with_zero_mean_and_unit_variance(image_bhwc)

                    alignment_transform = tf.Variable(alignment_transforms[image_index, :])

                    best_alignment_transform = tf.Variable(alignment_transform)


                    for steps in range(max_steps):
                        alignment_transform, guess_loss = align_image_training_step(
                            image_bhwc,
                            alignment_transform=alignment_transforms[image_index, :],
                            target_image=target_image,
                            mask_image=downsampled_mask_image,
                            learning_rate_for_lod=learning_rate_for_lod,
                            max_update_for_lod=max_update_for_lod
                        )    
                        
                        best_alignment_transform.assign(alignment_transform)
                        losses[image_index].assign(guess_loss[0]) 
                    
                    alignment_transforms[image_index, :].assign(best_alignment_transform)
            print(f'losses averaged {tf.reduce_mean(losses)}')
    return alignment_transforms

dark_image = None
if file_glob_darks is not None:
    dark_image, dark_image_count = util.load_average_image(file_glob_darks)

alignment_transforms = align_images(file_glob, dark_image)

final_average_image_bhwc = read_average_image_with_alignment_transforms(file_glob, alignment_transforms, image_shape, dark_image)
util.write_image(final_average_image_bhwc, os.path.join(output_path, 'final_average_image.png'))

image_index = -1
for filename in glob.glob(file_glob):
    for image_bhwc, frame_index in util.ImageSequenceReader(filename, step = frame_step, to_float = True, demosaic = True):
        image_index += 1

        # must be in the loop
        if dark_image is not None:
            image_bhwc -= dark_image

        alignment_transform = alignment_transforms[image_index, :]
        transformed_image_bhwc = align.transform_image(image_bhwc, alignment_transform)
        util.write_image(transformed_image_bhwc, os.path.join(output_path, "aligned_{:08d}.png".format(image_index)))