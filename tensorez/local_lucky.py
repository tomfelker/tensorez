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
import datetime
import sys
from tensorez.util import *
from tensorez.image_sequence import *
from tensorez.model import *
from tensorez.bayer import *
from tensorez.fourier import *
from tensorez.obd import *


def local_lucky(
    lights,
    crossover_wavelength_pixels = 30,
    noise_wavelength_pixels = 16,
    steepness = 3,
    stdevs_above_mean = 3,
    debug_output_dir = None,
    debug_frames = 10
):
    shape = hwc_to_chw(lights[0]).shape

    known_frequency_mask = get_gaussian_lowpass_frequency_mask(shape, crossover_wavelength_pixels)
    if debug_output_dir is not None:
        write_image(chw_to_hwc(known_frequency_mask), os.path.join(debug_output_dir, 'known_frequency_mask.png'))

    interesting_frequency_mask = get_gaussian_bandpass_frequency_mask(shape, noise_wavelength_pixels, crossover_wavelength_pixels)
    if debug_output_dir is not None:
        write_image(chw_to_hwc(interesting_frequency_mask), os.path.join(debug_output_dir, 'interesting_frequencies.png'))
    
    luckiness_mean, luckiness_stdev = pass_1(shape, lights, known_frequency_mask, interesting_frequency_mask, debug_output_dir, debug_frames)

    weighted_avg = pass_2(shape, lights, luckiness_mean, luckiness_stdev, known_frequency_mask, interesting_frequency_mask, stdevs_above_mean, steepness, debug_output_dir, debug_frames)

    return chw_to_hwc(weighted_avg)

def pass_1(shape, lights, known_frequency_mask, interesting_frequency_mask, debug_output_dir, debug_frames):
    luckiness_variance_state = welfords_init(shape)
    image_avg = tf.Variable(tf.zeros(shape))

    # todo: refactor these passes into functions for cleanliness / gc-ability

    # first pass
    for image_index, image in enumerate(lights):
        image = hwc_to_chw(image)

        image_avg.assign_add(image)

        luckiness = compute_luckiness(image, known_frequency_mask, interesting_frequency_mask)
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


def pass_2(shape, lights, luckiness_mean, luckiness_stdev, known_frequency_mask, interesting_frequency_mask, stdevs_above_mean, steepness, debug_output_dir, debug_frames):
    # second pass
    total_weight = tf.Variable(tf.zeros(shape))
    weighted_avg = tf.Variable(tf.zeros(shape))
    for image_index, image in enumerate(lights):
        image = hwc_to_chw(image)

        luckiness = compute_luckiness(image, known_frequency_mask, interesting_frequency_mask)
    
        luckiness_zero_mean_unit_variance = (luckiness - luckiness_mean) / luckiness_stdev

        if debug_output_dir is not None and image_index < debug_frames:
            write_image(chw_to_hwc(luckiness_zero_mean_unit_variance), os.path.join(debug_output_dir, "luckiness_zero_mean_unit_variance_{:08d}.png".format(image_index)), normalize = True)

        weight_image = tf.math.sigmoid((luckiness_zero_mean_unit_variance - stdevs_above_mean) * steepness)

        if debug_output_dir is not None and image_index < debug_frames:
            write_image(chw_to_hwc(weight_image), os.path.join(debug_output_dir, "luckiness_weights_{:08d}.png".format(image_index)))

        total_weight.assign_add(weight_image)
        weighted_avg.assign_add(weight_image * image)
        print(f"Pass 2 of 2, processed image {image_index + 1} of {len(lights)}")

    weighted_avg.assign(weighted_avg / total_weight)
    if debug_output_dir is not None:
        avg_num_frames = tf.reduce_mean(total_weight)
        write_image(chw_to_hwc(weighted_avg), os.path.join(debug_output_dir, f"local_lucky_{stdevs_above_mean}_stdevs_{steepness}_steepness_{int(avg_num_frames)}_of_{len(lights)}.png"))
        write_image(chw_to_hwc(total_weight), os.path.join(debug_output_dir, "total_weight.png"), normalize = True)

    return weighted_avg


@tf.function
def compute_luckiness(image_chw, known_frequency_mask, interesting_frequency_mask):

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
    spatial_domain_image_chw = tf.abs(spatial_domain_image_chw_complex)
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


