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

class LuckinessAlgorithmImageSquared:
    def create_cache(shape_chw, lights, average_image, debug_output_dir, debug_frames, isoplanatic_patch_pixels):
        isoplanatic_patch_frequency_mask = get_gaussian_lowpass_frequency_mask(shape_chw, isoplanatic_patch_pixels)
        cache = {}
        cache['isoplanatic_patch_frequency_mask'] = isoplanatic_patch_frequency_mask
        return cache

    @tf.function
    def compute_luckiness(image_chw, cache, isoplanatic_patch_pixels):
        isoplanatic_patch_frequency_mask = cache['isoplanatic_patch_frequency_mask']

        #image_chw -= tf.reduce_mean(image_chw, axis = [-2, -1], keepdims=True)

        luckiness_chw = image_chw * image_chw
        lowpass_luckiness_chw = apply_frequency_mask(luckiness_chw, isoplanatic_patch_frequency_mask)

        return lowpass_luckiness_chw


class LuckinessAlgorithmImageTimesKnown:
    def create_cache(shape_chw, lights, average_image, debug_output_dir, debug_frames, isoplanatic_patch_pixels):
        if average_image is not None:
            known_image_chw = hwc_to_chw(average_image)
        else:
            known_image_chw = tf.Variable(tf.zeros(shape_chw), dtype = tf.float32)
            for image_index, image in enumerate(lights):
                print(f"Averaging image {image_index + 1} of {len(lights)}")            
                known_image_chw.assign_add(hwc_to_chw(image))
                
            known_image_chw.assign(known_image_chw / len(lights))

        if debug_output_dir is not None:
            write_image(chw_to_hwc(known_image_chw), os.path.join(debug_output_dir, 'average_image.png'))

        #known_image_chw = known_image_chw - tf.reduce_mean(known_image_chw, axis = [-2, -1], keepdims=True)

        if debug_output_dir is not None:
            write_image(chw_to_hwc(known_image_chw), os.path.join(debug_output_dir, 'known_image.png'))

        isoplanatic_patch_frequency_mask = get_gaussian_lowpass_frequency_mask(shape_chw, isoplanatic_patch_pixels)

        lowpass_known_image_chw = apply_frequency_mask(known_image_chw, isoplanatic_patch_frequency_mask)

        cache = {}
        cache['known_image_chw'] = known_image_chw
        cache['lowpass_known_image_chw'] = lowpass_known_image_chw
        cache['isoplanatic_patch_frequency_mask'] = isoplanatic_patch_frequency_mask
        return cache

    @tf.function
    def compute_luckiness(image_chw, cache, isoplanatic_patch_pixels):
        known_image_chw = cache['known_image_chw']
        lowpass_known_image_chw = cache['lowpass_known_image_chw']
        isoplanatic_patch_frequency_mask = cache['isoplanatic_patch_frequency_mask']

        #image_chw -= tf.reduce_mean(image_chw, axis = [-2, -1], keepdims=True)

        luckiness_chw = image_chw * known_image_chw
        lowpass_luckiness_chw = apply_frequency_mask(luckiness_chw, isoplanatic_patch_frequency_mask)

        return lowpass_luckiness_chw / (lowpass_known_image_chw * lowpass_known_image_chw)

class LuckinessAlgorithmFrequencyBands:
    '''
    We want to select areas of images that:
        - give us new information about higher frequencies which aren't present in the average image
        - don't disagree with what we know about the lower frequencies, which are present in the average image

    
    '''
    def create_cache(shape_chw, lights, average_image, debug_output_dir, debug_frames, isoplanatic_patch_pixels, crossover_wavelength_pixels, noise_wavelength_pixels):

        isoplanatic_patch_frequency_mask = get_gaussian_lowpass_frequency_mask(shape_chw, isoplanatic_patch_pixels)
        known_frequency_mask = get_gaussian_bandpass_frequency_mask(shape_chw, crossover_wavelength_pixels, isoplanatic_patch_pixels)
        interesting_frequency_mask = get_gaussian_bandpass_frequency_mask(shape_chw, noise_wavelength_pixels, crossover_wavelength_pixels)

        # we do need the average image, but let's also do the FFT stuff, because it'd be cool to be able to automatically guess the wavelenghs.

        if average_image is not None:
            average_image_chw = hwc_to_chw(average_image)
        else:
            image_variance_state = welfords_init(shape_chw)
            frequency_mag_variance_state = welfords_init(shape_chw)
            
            for image_index, image in enumerate(lights):
                print(f"Averaging image {image_index + 1} of {len(lights)}")

                image = hwc_to_chw(image)
                image_variance_state = welfords_update(image_variance_state, image)

                frequency = spatial_to_frequency_domain(image)
                frequency_mag = tf.abs(frequency)
                frequency_mag_variance_state = welfords_update(frequency_mag_variance_state, frequency_mag)


            average_image_chw = welfords_get_mean(image_variance_state)
            # todo: do something cool with known image variance?  or else we could just average it, instead of Welford's, for a slight optimization,

            if debug_output_dir is not None:
                output_frequency_histograms(debug_output_dir, frequency_mag_variance_state, known_frequency_mask, interesting_frequency_mask)

        known_image_chw = apply_frequency_mask(average_image_chw, known_frequency_mask)

        if debug_output_dir is not None:
            write_image(chw_to_hwc(average_image_chw), os.path.join(debug_output_dir, 'average_image.png'))
            write_image(chw_to_hwc(known_image_chw), os.path.join(debug_output_dir, 'known_image.png'))
            write_image(chw_to_hwc(isoplanatic_patch_frequency_mask), os.path.join(debug_output_dir, 'isoplanatic_patch_frequency_mask.png'))
            write_image(chw_to_hwc(known_frequency_mask), os.path.join(debug_output_dir, 'known_frequency_mask.png'))
            write_image(chw_to_hwc(interesting_frequency_mask), os.path.join(debug_output_dir, 'interesting_frequency_mask.png'))
        
        cache = dict(
            isoplanatic_patch_frequency_mask = isoplanatic_patch_frequency_mask,
            known_frequency_mask = known_frequency_mask,
            interesting_frequency_mask = interesting_frequency_mask,
            known_image_chw = known_image_chw,
        )
        return cache

    @tf.function
    def compute_luckiness(image_chw, cache, isoplanatic_patch_pixels, crossover_wavelength_pixels, noise_wavelength_pixels):
        isoplanatic_patch_frequency_mask = cache['isoplanatic_patch_frequency_mask']
        known_frequency_mask = cache['known_frequency_mask']
        interesting_frequency_mask = cache['interesting_frequency_mask']
        known_image_chw = cache['known_image_chw']

        interesting_image_chw = apply_frequency_mask(image_chw, interesting_frequency_mask)
        redundant_image_chw = apply_frequency_mask(image_chw, known_frequency_mask)

        # abs or square?  abs kinda seems a little better maybe, and unfortunately i don't know any theory for which would be better
        # or maybe square + blur + sqrt?

        if True:
            if True:
                new_info_chw = tf.abs(interesting_image_chw - known_image_chw)
                wrong_info_chw = tf.abs(redundant_image_chw - known_image_chw)
            else:
                new_info_chw = tf.square(interesting_image_chw - known_image_chw)
                wrong_info_chw = tf.square(redundant_image_chw - known_image_chw)

            luckiness_chw = new_info_chw - wrong_info_chw
            lowpass_luckiness_chw = apply_frequency_mask(luckiness_chw, isoplanatic_patch_frequency_mask)

        if False:
            # or maybe square + blur + sqrt?
            new_info_chw = tf.square(interesting_image_chw - known_image_chw)
            wrong_info_chw = tf.square(redundant_image_chw - known_image_chw)

            new_info_chw = apply_frequency_mask(new_info_chw, isoplanatic_patch_frequency_mask)
            wrong_info_chw = apply_frequency_mask(wrong_info_chw, isoplanatic_patch_frequency_mask)
            lowpass_luckiness_chw = tf.sqrt(new_info_chw) - tf.sqrt(wrong_info_chw)

        return lowpass_luckiness_chw


class LuckinessAlgorithmLowpassAbsBandpass:
    def create_cache(shape, lights, average_image, debug_output_dir, debug_frames, crossover_wavelength_pixels, noise_wavelength_pixels):
        known_frequency_mask = get_gaussian_lowpass_frequency_mask(shape, crossover_wavelength_pixels)
        if debug_output_dir is not None:
            write_image(chw_to_hwc(known_frequency_mask), os.path.join(debug_output_dir, 'known_frequency_mask.png'))

        interesting_frequency_mask = get_gaussian_bandpass_frequency_mask(shape, noise_wavelength_pixels, crossover_wavelength_pixels)
        if debug_output_dir is not None:
            write_image(chw_to_hwc(interesting_frequency_mask), os.path.join(debug_output_dir, 'interesting_frequencies.png'))

        # this stuff isn't necessary for the algorithm, just nice to see
        if debug_output_dir is not None:
            frequency_mag_variance_state = welfords_init(shape)

            for image_index, image in enumerate(lights):
                if image_index >= debug_frames:
                    break

                image = hwc_to_chw(image)

                frequency = spatial_to_frequency_domain(image)
                frequency_mag = tf.abs(frequency)
                frequency_mag_variance_state = welfords_update(frequency_mag_variance_state, frequency_mag)

            output_frequency_histograms(debug_output_dir, frequency_mag_variance_state, known_frequency_mask, interesting_frequency_mask)

        cache = {}
        cache['known_frequency_mask'] = known_frequency_mask
        cache['interesting_frequency_mask'] = interesting_frequency_mask
        return cache

    @tf.function
    def compute_luckiness(image_chw, cache, crossover_wavelength_pixels, noise_wavelength_pixels):
        known_frequency_mask = cache['known_frequency_mask']
        interesting_frequency_mask = cache['interesting_frequency_mask']

        highpass_image = apply_frequency_mask(image_chw, interesting_frequency_mask)
        # In theory, if we then take the lowpass, we'll just get zero... but that's just because
        # our image looks 0 from across the room, since what was a sharp rising edge from 0 to 1 has become
        # a gentle fall to -0.5, a sharp rise to 0.5, and a gentle fall back to 0, so it all cancels out.
        # But if we take the abs, or square it, it's now a spike above zero.

        # Not sure whether this should be abs() or square(), but it should probably match
        # what we did to agreement_image above.
        edgey_image = tf.abs(highpass_image)

        luckiness = apply_frequency_mask(edgey_image, known_frequency_mask)
        return luckiness




def get_frequency_bins(shape_chw):
    x = tf.expand_dims(tf.minimum(tf.range(0.0, shape_chw[-1], 1), tf.range(shape_chw[-1] - 1.0, -1, -1)), -2)
    y = tf.expand_dims(tf.minimum(tf.range(0.0, shape_chw[-2], 1), tf.range(shape_chw[-2] - 1.0, -1, -1)), -1)
    dist_from_corner = tf.sqrt(tf.square(x) + tf.square(y))
    dist_from_corner = tf.cast(dist_from_corner, dtype=tf.int32)
    dist_from_corner = tf.expand_dims(dist_from_corner, axis = -3)
    dist_from_corner = tf.expand_dims(dist_from_corner, axis = -4)
    return dist_from_corner

def write_hist(name, data, bins, bins_hist, num_bins, debug_output_dir):
    hist = tf.math.divide_no_nan(bincount_along_axis(bins, weights = data, length=num_bins, axis = -3), bins_hist)
    hist = hist / tf.reduce_max(hist)
    hist_graph = vector_per_channel_to_graph(hist)
    write_image(hist_graph, os.path.join(debug_output_dir, f'{name}.png'))
    return hist_graph


def pass_1(shape, lights, algorithm, algorithm_cache, algorithm_kwargs, debug_output_dir, debug_frames):
    luckiness_variance_state = welfords_init(shape)
    image_avg = tf.Variable(tf.zeros(shape))

    # first pass
    for image_index, image in enumerate(lights):
        image = hwc_to_chw(image)

        image_avg.assign_add(image)

        luckiness = algorithm.compute_luckiness(image, algorithm_cache, **algorithm_kwargs)
        if debug_output_dir is not None and image_index < debug_frames:
            write_image(chw_to_hwc(luckiness), os.path.join(debug_output_dir, "luckiness_{:08d}.png".format(image_index)), normalize = True)

        luckiness_variance_state = welfords_update(luckiness_variance_state, luckiness)
        print(f"Pass 1 of 2, processed image {image_index + 1} of {len(lights)}")

    image_avg.assign(image_avg / len(lights))

    if debug_output_dir is not None:
        write_image(chw_to_hwc(image_avg), os.path.join(debug_output_dir, 'unweighted_avg.png'))

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

        luckiness = algorithm.compute_luckiness(image, algorithm_cache, **algorithm_kwargs)

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

    
def spatial_to_frequency_domain(spatial_domain_image_chw):
    spatial_domain_image_chw_complex = tf.cast(spatial_domain_image_chw, dtype = tf.complex64)
    frequency_domain_image_chw_complex = tf.signal.fft2d(spatial_domain_image_chw_complex)
    return frequency_domain_image_chw_complex

def frequency_to_spatial_domain(frequency_domain_image_chw):
    spatial_domain_image_chw_complex = tf.signal.ifft2d(frequency_domain_image_chw)
    spatial_domain_image_chw = tf.math.real(spatial_domain_image_chw_complex)
    return spatial_domain_image_chw

def apply_frequency_mask(image_chw, mask_chw):
    mask_chw_complex = tf.cast(mask_chw, tf.complex64)
    return frequency_to_spatial_domain(spatial_to_frequency_domain(image_chw) * mask_chw_complex)

def fft_spatial_frequencies_per_pixel(shape_chw):
    w = shape_chw[-1]
    h = shape_chw[-2]
    freqs_x = tf.convert_to_tensor(np.fft.fftfreq(w), dtype = tf.float32)
    freqs_y = tf.convert_to_tensor(np.fft.fftfreq(h), dtype = tf.float32)
    freqs_x = tf.expand_dims(freqs_x, axis = [-2])
    freqs_y = tf.expand_dims(freqs_y, axis = [-1])
    freqs = tf.sqrt(freqs_x * freqs_x + freqs_y * freqs_y)
    return freqs

def get_highpass_frequency_mask(shape_chw, cutoff_wavelength_pixels):
    freqs = fft_spatial_frequencies_per_pixel(shape_chw)
    highpass = tf.greater(freqs, 1.0 / cutoff_wavelength_pixels)
    highpass = tf.cast(highpass, dtype = tf.float32)
    highpass = tf.expand_dims(highpass, axis = [-3])
    return highpass

def get_lowpass_frequency_mask(shape_chw, cutoff_wavelength_pixels):
    freqs = fft_spatial_frequencies_per_pixel(shape_chw)
    lowpass = tf.less(freqs, 1.0 / cutoff_wavelength_pixels)
    lowpass = tf.cast(lowpass, dtype = tf.float32)
    lowpass = tf.expand_dims(lowpass, axis = [-3])
    return lowpass

def get_bandpass_frequency_mask(shape_chw, min_wavelength_pixels, max_wavelength_pixels):
    return get_highpass_frequency_mask(shape_chw, max_wavelength_pixels) * get_lowpass_frequency_mask(shape_chw, min_wavelength_pixels)

def get_gaussian_lowpass_frequency_mask(shape_chw, cutoff_wavelength_pixels):
    freqs = fft_spatial_frequencies_per_pixel(shape_chw)
    lowpass = gaussian(freqs, mean = 0, variance = tf.square(1.0 / cutoff_wavelength_pixels))
    lowpass /= gaussian(0, mean = 0, variance = tf.square(1.0 / cutoff_wavelength_pixels))
    lowpass = tf.expand_dims(lowpass, axis = [-3])
    return lowpass

def get_gaussian_highpass_frequency_mask(shape_chw, cutoff_wavelength_pixels):
    return 1 - get_gaussian_lowpass_frequency_mask(shape_chw, cutoff_wavelength_pixels)

def get_gaussian_bandpass_frequency_mask(shape_chw, min_wavelength_pixels, max_wavelength_pixels):
    return get_gaussian_highpass_frequency_mask(shape_chw, max_wavelength_pixels) * get_gaussian_lowpass_frequency_mask(shape_chw, min_wavelength_pixels)



# Straight from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm.
# Interestingly, the very functional-programming style they used is perfect for avoiding tf.function weirdness.

def welfords_init(shape):
    # hmm, I can never tell if I should use tf.variable or trust in the tracing magic
    count = tf.zeros([])
    mean = tf.zeros(shape)
    M2 = tf.zeros(shape)
    return (count, mean, M2)

@tf.function
def welfords_update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

def welfords_get_mean(existingAggregate):
    (count, mean, M2) = existingAggregate
    return mean

def welfords_get_variance(existingAggregate):
    (count, mean, M2) = existingAggregate
    return M2 / count

def welfords_get_stdev(existingAggregate):
    (count, mean, M2) = existingAggregate
    return tf.sqrt(M2 / count)

def bincount_along_axis(bins, weights, length, axis):
    # todo: if this didn't work, we'd need to unstack both
    bins = tf.squeeze(bins, [axis])

    bincounts = []
    for weights_slice in tf.unstack(weights, axis=axis):
        bincounts.append(tf.math.bincount(bins, weights=weights_slice, minlength=length, maxlength=length))
    # everything will have been flattened
    stacked = tf.stack(bincounts)
    return stacked

def output_frequency_histograms(debug_output_dir, frequency_mag_variance_state, known_frequency_mask, interesting_frequency_mask):
    frequency_mag_mean = welfords_get_mean(frequency_mag_variance_state)
    frequency_mag_stdev = welfords_get_stdev(frequency_mag_variance_state)

    # discard channel info    
    #frequency_mag_mean = tf.reduce_mean(frequency_mag_mean, axis = -3, keepdims=True)
    #frequency_mag_stdev = tf.reduce_mean(frequency_mag_stdev, axis = -3, keepdims=True)

    shape = frequency_mag_mean.shape
    bins = get_frequency_bins(shape)

    num_bins = tf.maximum(shape[-1], shape[-2])

    bins_hist = tf.math.bincount(bins, minlength=num_bins, maxlength=num_bins)
    bins_hist = tf.cast(bins_hist, dtype=tf.float32)

    frequency_mag_hist = tf.math.divide_no_nan(bincount_along_axis(bins, weights = frequency_mag_mean, length=num_bins, axis = -3), bins_hist)
    frequency_mag_hist_hi = tf.math.divide_no_nan(bincount_along_axis(bins, weights = frequency_mag_mean + frequency_mag_stdev, length=num_bins, axis = -3), bins_hist)
    frequency_mag_hist_lo = tf.math.divide_no_nan(bincount_along_axis(bins, weights = frequency_mag_mean - frequency_mag_stdev, length=num_bins, axis = -3), bins_hist)
    frequency_stdev_hist = tf.math.divide_no_nan(bincount_along_axis(bins, weights = frequency_mag_stdev, length=num_bins, axis = -3), bins_hist)
    frequency_mag_uncertainty_hist = tf.math.divide_no_nan(frequency_stdev_hist, frequency_mag_hist)

    xmax = tf.reduce_max(frequency_mag_hist_hi)
    frequency_mag_hist /= xmax
    frequency_mag_hist_hi /= xmax
    frequency_mag_hist_lo /= xmax

    red = tf.reshape(tf.constant([1.0, 0, 0]), shape=[1, 1, 1, 3])
    green = tf.reshape(tf.constant([0, 1.0, 0]), shape=[1, 1, 1, 3])
    blue = tf.reshape(tf.constant([0, 0, 1.0]), shape=[1, 1, 1, 3])

    frequency_mag_hist_chart = vector_per_channel_to_graph(frequency_mag_hist)
    frequency_mag_hist_chart += vector_per_channel_to_graph(frequency_mag_hist_hi)
    frequency_mag_hist_chart += vector_per_channel_to_graph(frequency_mag_hist_lo)
    write_image(frequency_mag_hist_chart, os.path.join(debug_output_dir, 'histogram_frequency_mag.png'))

    frequency_mag_uncertainty_hist /= tf.reduce_max(frequency_mag_uncertainty_hist)
    frequency_mag_uncertainty_hist_chart = vector_per_channel_to_graph(frequency_mag_uncertainty_hist)
    write_image(frequency_mag_uncertainty_hist_chart, os.path.join(debug_output_dir, 'histogram_frequency_mag_uncertainty.png'))

    histogram_lucky_known = write_hist('histogram_lucky_known', tf.expand_dims(known_frequency_mask, axis = -4), bins, bins_hist, num_bins, debug_output_dir)
    histogram_lucky_interesting = write_hist('histogram_lucky_interesting', tf.expand_dims(interesting_frequency_mask, axis = -4), bins, bins_hist, num_bins, debug_output_dir)

    write_hist('histogram_mean', frequency_mag_mean, bins, bins_hist, num_bins, debug_output_dir)
    write_hist('histogram_stdev', frequency_mag_stdev, bins, bins_hist, num_bins, debug_output_dir)
    histogram_stdev_over_mean = write_hist('histogram_stdev_over_mean', frequency_mag_stdev / frequency_mag_mean, bins, bins_hist, num_bins, debug_output_dir)

    combo = histogram_lucky_known * (red + blue) + histogram_lucky_interesting * (green + blue) + histogram_stdev_over_mean
    write_image(combo, os.path.join(debug_output_dir, 'histogram_combo.png'))
