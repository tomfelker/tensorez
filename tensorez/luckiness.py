import numpy as np
import tensorflow as tf
from tensorez.util import *

class LuckinessAlgorithmImageSquared:
    def create_cache(shape_chw, lights, average_image, debug_output_dir, debug_frames, isoplanatic_patch_pixels):
        isoplanatic_patch_frequency_mask = get_gaussian_lowpass_frequency_mask(shape_chw, isoplanatic_patch_pixels)
        cache = {}
        cache['isoplanatic_patch_frequency_mask'] = isoplanatic_patch_frequency_mask
        return cache

    @tf.function
    def compute_luckiness(image_chw, cache, want_debug_images, isoplanatic_patch_pixels):
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
    def compute_luckiness(image_chw, cache, want_debug_images, isoplanatic_patch_pixels):
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
    def create_cache(shape_chw, lights, average_image, debug_output_dir, debug_frames, isoplanatic_patch_pixels, crossover_wavelength_pixels, noise_wavelength_pixels, misalignment_penalty_factor = 5):

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

        average_image_known_chw = apply_frequency_mask(average_image_chw, known_frequency_mask)
        
        if debug_output_dir is not None:
            write_image(chw_to_hwc(average_image_chw), os.path.join(debug_output_dir, 'average_image.png'))
            write_image(chw_to_hwc(average_image_known_chw), os.path.join(debug_output_dir, 'average_image_known_chw.png'))

            average_image_interesting_chw = apply_frequency_mask(average_image_chw, interesting_frequency_mask)
            write_image(chw_to_hwc(average_image_interesting_chw), os.path.join(debug_output_dir, 'average_image_interesting_chw.png'))
            write_image(chw_to_hwc(isoplanatic_patch_frequency_mask), os.path.join(debug_output_dir, 'isoplanatic_patch_frequency_mask.png'))
            write_image(chw_to_hwc(known_frequency_mask), os.path.join(debug_output_dir, 'known_frequency_mask.png'))
            write_image(chw_to_hwc(interesting_frequency_mask), os.path.join(debug_output_dir, 'interesting_frequency_mask.png'))
        
        cache = dict(
            isoplanatic_patch_frequency_mask=isoplanatic_patch_frequency_mask,
            known_frequency_mask=known_frequency_mask,
            interesting_frequency_mask=interesting_frequency_mask,
            average_image_known_chw=average_image_known_chw,
        )
        return cache

    @tf.function
    def compute_luckiness(image_chw, cache, want_debug_images, isoplanatic_patch_pixels, crossover_wavelength_pixels, noise_wavelength_pixels, epsilon = 1.0 / (1 << 16)):
        isoplanatic_patch_frequency_mask = cache['isoplanatic_patch_frequency_mask']
        known_frequency_mask = cache['known_frequency_mask']
        interesting_frequency_mask = cache['interesting_frequency_mask']
        average_image_known_chw = cache['average_image_known_chw']

        this_image_known_chw = apply_frequency_mask(image_chw, known_frequency_mask)
        this_image_interesting_chw = apply_frequency_mask(image_chw, interesting_frequency_mask)
        
        new_info_chw = tf.square(this_image_interesting_chw)
        wrong_info_chw = tf.square(this_image_known_chw - average_image_known_chw)

        new_info_lowpass_chw = apply_frequency_mask(new_info_chw, isoplanatic_patch_frequency_mask)
        wrong_info_lowpass_chw = apply_frequency_mask(wrong_info_chw, isoplanatic_patch_frequency_mask)

        lowpass_luckiness_chw = tf.sqrt(tf.math.divide_no_nan(new_info_lowpass_chw, (wrong_info_lowpass_chw + epsilon)))

        debug_images = {}
        if want_debug_images:
            debug_images['new_info'] = new_info_chw
            debug_images['wrong_info'] = wrong_info_chw

        return lowpass_luckiness_chw, debug_images


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
    def compute_luckiness(image_chw, cache, want_debug_images, crossover_wavelength_pixels, noise_wavelength_pixels):
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



def bincount_along_axis(bins, weights, length, axis):
    # todo: if this didn't work, we'd need to unstack both
    bins = tf.squeeze(bins, [axis])

    bincounts = []
    for weights_slice in tf.unstack(weights, axis=axis):
        bincounts.append(tf.math.bincount(bins, weights=weights_slice, minlength=length, maxlength=length))
    # everything will have been flattened
    stacked = tf.stack(bincounts)
    return stacked
