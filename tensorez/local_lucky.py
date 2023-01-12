"""
Local Lucky Imaging

The idea is to apply lucky imaging on a per-pixel rather than per-frame basis.

The quality of the image varies not only in time, but across space, so it's
unlikely that the entire frame will be sharp at the same time - this limits
traditional lucky imaging, because that involves choosing entire frames.

Instead, we will try to determine "luckiness" for each pixel of each frame,
and then for each pixel in the result, average the corresponding luckiest
pixels in the input.

A lucky frame should agree with the average frame at low frequencies, but
contain more information than the average frame at high frequencies.  We can
use some FFT magic and hand waving to compute this.

To strictly choose from the best N frames, we would need to remember N
luckinesses and frame indices, which may be too memory intensive.  We can work
around that by doing two passes, the first pass computing luckinesses and their
per-pixel mean and variance, and the second pass re-computing the luckiness and
computing a weighted average of the pixel values, with high weights for pixels
luckier than a threshold - in other words, "average of all pixels more than
two standard deviations luckier than the mean".  This does assume a Gaussian
distribution of luckiness.
"""


import numpy as np
import tensorflow as tf
import os
from tensorez.util import *
from tensorez.image_sequence import *
from tensorez.model import *
from tensorez.bayer import *
from tensorez.fourier import *
from tensorez.obd import *
import tensorez.drizzle
import tensorez.align
import tensorez.modified_stn
from tensorez.luckiness import *

def local_lucky(
    lights,
    algorithm,
    algorithm_kwargs,
    average_image = None,
    steepness = 3,
    stdevs_above_mean = 3,
    debug_output_dir = None,
    debug_frames = 10,
    drizzle = True,
    drizzle_kwargs = {},
    bayer = True
):
    shape = hwc_to_chw(lights[0]).shape

    algorithm_cache = algorithm.create_cache(shape, lights, average_image, debug_output_dir, debug_frames, **algorithm_kwargs)

    luckiness_mean, luckiness_stdev = pass_1(shape, lights, algorithm, algorithm_cache, algorithm_kwargs, debug_output_dir, debug_frames)

    weighted_avg = pass_2(lights, luckiness_mean, luckiness_stdev, algorithm, algorithm_cache, algorithm_kwargs, stdevs_above_mean, steepness, debug_output_dir, debug_frames, drizzle, drizzle_kwargs, bayer)

    return chw_to_hwc(weighted_avg)






def pass_1(shape, lights, algorithm, algorithm_cache, algorithm_kwargs, debug_output_dir, debug_frames):
    luckiness_variance_state = welfords_init(shape)

    if debug_output_dir is not None:
        image_avg = tf.Variable(tf.zeros(shape))

    # first pass
    for image_index, image in enumerate(lights):
        image = hwc_to_chw(image)

        if debug_output_dir is not None:
            image_avg.assign_add(image)

        want_debug_images = (debug_output_dir is not None and image_index < debug_frames)

        luckiness, debug_images = algorithm.compute_luckiness(image, algorithm_cache, want_debug_images, **algorithm_kwargs)

        if want_debug_images:
            write_image(chw_to_hwc(luckiness), os.path.join(debug_output_dir, "luckiness_{:08d}.png".format(image_index)), normalize = True)
            for debug_image_name, debug_image in debug_images.items():
                write_image(chw_to_hwc(debug_image), os.path.join(debug_output_dir, f"{debug_image_name}_{image_index:08d}.png"))


        luckiness_variance_state = welfords_update(luckiness_variance_state, luckiness)
        print(f"Pass 1 of 2, processed image {image_index + 1} of {len(lights)}")

    if debug_output_dir is not None:
        image_avg.assign(image_avg / len(lights))
        write_image(chw_to_hwc(image_avg), os.path.join(debug_output_dir, 'local_lucky_unweighted_average.png'))

    luckiness_mean = welfords_get_mean(luckiness_variance_state)
    luckiness_stdev = welfords_get_stdev(luckiness_variance_state)

    return luckiness_mean, luckiness_stdev


def pass_2(lights, luckiness_mean, luckiness_stdev, algorithm, algorithm_cache, algorithm_kwargs, stdevs_above_mean, steepness, debug_output_dir, debug_frames, drizzle, drizzle_kwargs, bayer):
    # second pass
    if bayer:
        bayer_filter = lights.read_bayer_filter_unaligned()

        #if debug_output_dir is not None:
        #    write_image(bayer_filter, os.path.join(debug_output_dir, "bayer_filter.png"))

    total_weight = None
    weighted_avg = None
    for image_index, image in enumerate(lights):
        image = hwc_to_chw(image)

        luckiness, debug_images = algorithm.compute_luckiness(image, algorithm_cache, want_debug_images = False, **algorithm_kwargs)

        luckiness_zero_mean_unit_variance = tf.math.divide_no_nan(luckiness - luckiness_mean, luckiness_stdev)

        if debug_output_dir is not None and image_index < debug_frames:
            write_image(chw_to_hwc(luckiness_zero_mean_unit_variance), os.path.join(debug_output_dir, "luckiness_zero_mean_unit_variance_{:08d}.png".format(image_index)), normalize = True)
        
        weight_image = tf.math.sigmoid((luckiness_zero_mean_unit_variance - stdevs_above_mean) * steepness)

        # in these cases, we have input space weights (the drizzle mask, or the bayer mask, or todo dead pixel stuff), so we need
        # to align those (and possibly also align the image)
        if drizzle or bayer:
            raw_image_bhwc = lights.read_cooked_image(image_index, skip_content_align = True)
            alignment_transform = lights.alignment_transforms[image_index]
            theta = tensorez.align.alignment_transform_to_stn_theta(alignment_transform)

            if bayer:
                inputspace_weights_bhwc = bayer_filter
            else:
                inputspace_weights_bhwc = None
                # todo: dead pixel stuff, variance of darks, also as an input weight

            if drizzle:
                # note: this replaces image, above, which was read with the STN based alignment, with a possibly bigger one.
                image, aligned_weights = tensorez.drizzle.drizzle(raw_image_bhwc, theta, inputspace_weights_bhwc = inputspace_weights_bhwc, **drizzle_kwargs, )
                image = hwc_to_chw(image)
                aligned_weights = hwc_to_chw(aligned_weights)

                # drizzle probably made things bigger, so need to upscale the weights too
                if weight_image.shape != image.shape:
                    weight_image = hwc_to_chw(tensorez.modified_stn.spatial_transformer_network(chw_to_hwc(weight_image), tf.eye(2, 3), out_dims = (image.shape[-2], image.shape[-1])))

            else:
                assert(inputspace_weights_bhwc is not None)
                aligned_weights = hwc_to_chw(tensorez.modified_stn.spatial_transformer_network(inputspace_weights_bhwc, theta))

            weight_image *= aligned_weights

        if debug_output_dir is not None and image_index < debug_frames:
            write_image(chw_to_hwc(weight_image), os.path.join(debug_output_dir, "luckiness_weights_{:08d}.png".format(image_index)))
            write_image(chw_to_hwc(weight_image), os.path.join(debug_output_dir, "luckiness_weights_normalized{:08d}.png".format(image_index)), normalize=True)


        if total_weight is None:
            total_weight = tf.Variable(tf.zeros_like(weight_image))
        if weighted_avg is None:
            weighted_avg = tf.Variable(tf.zeros_like(weight_image))

        total_weight.assign_add(weight_image)
        weighted_avg.assign_add(weight_image * image)
        print(f"Pass 2 of 2, processed image {image_index + 1} of {len(lights)}")

    weighted_avg.assign(tf.math.divide_no_nan(weighted_avg, total_weight))
    if debug_output_dir is not None:
        avg_num_frames = tf.reduce_mean(total_weight)
        write_image(chw_to_hwc(weighted_avg), os.path.join(debug_output_dir, f"local_lucky_{stdevs_above_mean}_stdevs_{steepness}_steepness_{int(avg_num_frames)}_of_{len(lights)}.png"))
        write_image(chw_to_hwc(total_weight), os.path.join(debug_output_dir, "total_weight.png"), normalize = True)

    return weighted_avg

