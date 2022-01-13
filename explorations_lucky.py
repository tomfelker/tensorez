"""
Lucky Imaging - or that was the idea, but while looking at how to score the frames, I stumbled upon this paper:

Garrel, Vincent, Olivier Guyon, and Pierre Baudoz. “A Highly Efficient Lucky Imaging Algorithm: Image Synthesis Based on Fourier Amplitude Selection.” Publications of the Astronomical Society of the Pacific 124, no. 918 (2012): 861–67. https://doi.org/10.1086/667399.
https://www.jstor.org/stable/10.1086/667399
https://doi.org/10.1086/667399

and it was even easier to implement than lucky imaging.  And this is the first result I'm truly happy with...

Currently I'm just taking the one max value, instead of averaging over all the values over some quantile.
Doing that would take more memory... but!  It seems nicer to user SoftMax instead, so I'll just do that.

Also still todo, blurring the FFTs to reduce effects of photon noise.

"""


import numpy as np
import tensorflow as tf
import os
import glob
import datetime
import gc
import sys
import time
import string
import math
from tensorez.util import *
from tensorez.model import *
from tensorez.bayer import *
from tensorez.fourier import *
from tensorez.obd import *

###############################################################################
# Settings
align_by_center_of_mass = True
file_glob_darks = None
crop = None

only_even_shifts = False
max_align_steps = 50

light_frame_limit = None
light_frame_skip = None
dark_frame_limit = None
debug_frame_limit = 10

softmax_temperature = 1
do_blur = True
also_blur_phase = True
blur_standard_deviation = 4


###############################################################################
# Data selection

#file_glob = os.path.join('data', '2021-10-15_jupiter_prime', 'jupiter*.light.SER'); file_glob_darks = os.path.join('data', '2021-10-15_jupiter_prime', 'jupiter*.dark.SER'); crop = (2048, 2048)
#file_glob = os.path.join('data', '2021-10-15_jupiter_prime_crop', '*.light.SER'); file_glob_darks = os.path.join('data', '2021-10-15_jupiter_prime_crop', '*.dark.SER'); crop = (512, 512)
#file_glob = os.path.join('data', '2021-10-15_jupiter_barlow3x', '2021-10-16-0501_0-CapObj.SER'); file_glob_darks = os.path.join('data', '2021-10-15_jupiter_barlow3x', '2021-10-16-0506_5-CapObj.SER');
#file_glob = os.path.join('data', '2021-10-15_saturn_prime_crop', '2021-10-16-0445_8-CapObj.SER'); file_glob_darks = os.path.join('data', '2021-10-15_saturn_prime_crop', '2021-10-16-0448_0-CapObj.SER');
#file_glob = os.path.join('data', '2021-10-15_jupiter_prime', 'fake_jupiter.png');
#file_glob = os.path.join('data', 'ISS_aligned_from_The_8_Bit_Zombie', '*.tif')
file_glob = os.path.join('data', '2020-11-21Z_iss', '2020-11-21-0222_4-CapObj.SER'); light_frame_limit = 1000; light_frame_skip = 2900
###############################################################################
# Code

output_dir = os.path.join("output", "latest_lucky", datetime.datetime.now().replace(microsecond = 0).isoformat().replace(':', '_'))

os.makedirs(output_dir, exist_ok = True)
try:    
    import tee    
    tee.StdoutTee(os.path.join(output_dir, 'log.txt'), buff = 1).__enter__()
    sys.stderr = sys.stdout
except ModuleNotFoundError:
    print("Warning: to generate log.txt, need to install tee.")
    pass

@tf.function
def process_image(image_hwc, align_by_center_of_mass, crop):
    # todo: this is dumb, replace with real alignment
    # todo: it also probably works best if images are mean 0, could do that...
    if align_by_center_of_mass:
        for align_step in tf.range(max_align_steps):
            image_hwc, shift, shift_axis = center_image(image_hwc, only_even_shifts = only_even_shifts)
            #tf.print("Shifted by ", shift)
            if tf.reduce_max(tf.abs(shift)) < (2 if only_even_shifts else 1):
                #tf.print("Centered after ", align_step, " steps.")
                break
            if align_step + 1 >= max_align_steps:
                tf.print("Alignment didn't converge after ", align_step + 1, " steps.")
                break
    
    if crop is not None:
        image_hwc = crop_image(image_hwc, crop, crop_align = 1)

    return image_hwc        

dark_image = None
if file_glob_darks is not None:
    dark_image, image_count = load_average_image(file_glob_darks, frame_limit = dark_frame_limit)
    write_image(dark_image, os.path.join(output_dir, 'dark.png'))
    write_image(dark_image, os.path.join(output_dir, 'dark_normalized.png'), normalize = True)


def mag_for_softmax(image_fft_chw, blur_tf):
    if do_blur:
        if also_blur_phase:
            if False:
                # hmm, this doesn't seemt to work...
                # image_fft_blurred_chw = tf.signal.ifft2d(tf.signal.fft2d(image_fft_chw) * blur_tf)            
                # image_fft_mag_chw = tf.abs(image_fft_blurred_chw)
                # maybe blur real and imag separately?
                image_fft_blurred_real_chw = tf.abs(tf.signal.ifft2d(tf.signal.fft2d(tf.dtypes.complex(tf.math.real(image_fft_chw), tf.zeros_like(tf.math.real(image_fft_chw)))) * blur_tf))
                image_fft_blurred_imag_chw = tf.abs(tf.signal.ifft2d(tf.signal.fft2d(tf.dtypes.complex(tf.math.imag(image_fft_chw), tf.zeros_like(tf.math.imag(image_fft_chw)))) * blur_tf))
                image_fft_blurred_chw = tf.dtypes.complex(image_fft_blurred_real_chw, image_fft_blurred_imag_chw)
                image_fft_blurred_mag_chw = tf.abs(image_fft_blurred_chw)
                #return image_fft_blurred_mag_chw
                image_fft_mag_chw = tf.abs(image_fft_chw)
                image_fft_mag_blurred_chw = tf.abs(tf.signal.ifft2d(tf.signal.fft2d(tf.dtypes.complex(image_fft_mag_chw, tf.zeros_like(image_fft_mag_chw))) * blur_tf))
                spatial_correlation = tf.math.divide_no_nan(image_fft_blurred_mag_chw, image_fft_mag_blurred_chw)
                return spatial_correlation
            else:
                # hmm, just normalize then blur
                image_fft_mag = tf.math.abs(image_fft_chw)
                image_fft_phase = image_fft_chw * tf.dtypes.complex(tf.math.reciprocal_no_nan(image_fft_mag), tf.zeros_like(image_fft_mag))
                image_fft_phase_blurred_chw = tf.signal.ifft2d(tf.signal.fft2d(image_fft_phase) * blur_tf)
                return tf.abs(image_fft_phase_blurred_chw)
        else:
            image_fft_mag_chw = tf.abs(image_fft_chw)
            image_fft_mag_blurred_chw = tf.abs(tf.signal.ifft2d(tf.signal.fft2d(tf.dtypes.complex(image_fft_mag_chw, tf.zeros_like(image_fft_mag_chw))) * blur_tf))
            return image_fft_mag_blurred_chw
    else:
        return tf.abs(image_fft_chw)

# First, we do a pass to compute the elementwise maximum magnitude of the FFT of each frame - this is necessary
# for numerical stability of the softmax.  We can also do some nice-to-have things in this pass, like computing
# a simple average.  TODO: store off the centers of mass, for faster processing later.

@tf.function
def first_pass_step(average_chw, fft_mag_max_chw, image_chw, blur_tf):
    average_chw.assign(average_chw + image_chw)

    image_complex_chw = tf.dtypes.complex(image_chw, tf.zeros_like(image_chw))
    image_fft_chw = tf.signal.fft2d(image_complex_chw)        

    image_fft_mag_chw = mag_for_softmax(image_fft_chw, blur_tf)
    
    fft_mag_max_chw.assign(tf.maximum(fft_mag_max_chw, image_fft_mag_chw))

#have to fill this in later, once we've loaded an image (so we know the shape)
blur_tf = None

image_count = 0
average_chw = None
fft_mag_max_chw = None
for filename in glob.glob(file_glob):
    for image_hwc, frame_index in ImageSequenceReader(filename, skip = light_frame_skip, to_float = True, demosaic = True):           
        image_count += 1

        if dark_image is not None:
            image_hwc -= dark_image

        image_hwc = process_image(image_hwc, align_by_center_of_mass = align_by_center_of_mass, crop = crop)

        image_chw = hwc_to_chw(image_hwc)

        if average_chw is None:
            average_chw = tf.Variable(tf.zeros_like(image_chw))
            fft_mag_max_chw = tf.Variable(tf.zeros_like(image_chw))
            if do_blur:
                blur_kernel = gaussian_kernel_2d([image_chw.shape[-2],image_chw.shape[-1]], tf.constant(blur_standard_deviation, dtype = tf.float32))
                blur_kernel = blur_kernel / tf.reduce_sum(blur_kernel)
                blur_kernel = tf.signal.fftshift(blur_kernel)
                blur_kernel = tf.expand_dims(blur_kernel, axis = -3)

                write_image(chw_to_hwc(blur_kernel), os.path.join(output_dir, 'blur_kernel.png'), normalize = True)

                blur_tf = tf.signal.fft2d(tf.dtypes.complex(blur_kernel, tf.zeros_like(blur_kernel)))

                write_image(blur_tf, os.path.join(output_dir, 'blur_tf.png'), frequency_domain = True)

        first_pass_step(average_chw, fft_mag_max_chw, image_chw, blur_tf)
                
        if image_count <= debug_frame_limit:            
            write_image(tf.signal.fft2d(tf.dtypes.complex(image_chw, tf.zeros_like(image_chw))), os.path.join(output_dir, f'image_{image_count:08d}.png'), frequency_domain = True)            
            write_image(fft_mag_max_chw, os.path.join(output_dir, f'running_fft_mag_max_{image_count:08d}.png'), frequency_domain = True)
        
        if light_frame_limit is not None and image_count >= light_frame_limit:
            break

if image_count == 0:
    raise RuntimeError(f"Couldn't load any images from '{file_glob}'.")      

average_chw.assign(average_chw * (1.0 / image_count))

write_image(chw_to_hwc(average_chw), os.path.join(output_dir, 'average.png'), normalize = True)
write_image(tf.signal.fft2d(tf.dtypes.complex(average_chw, tf.zeros_like(average_chw))), os.path.join(output_dir, 'fft_average.png'), frequency_domain = True)

write_image(tf.dtypes.complex(fft_mag_max_chw, tf.zeros_like(fft_mag_max_chw)), os.path.join(output_dir, 'fft_mag_max.png'), frequency_domain = True)

# Next pass, we compute the softmax.

@tf.function
def second_pass_step(best_fft_mag_sum_chw, best_fft_chw, image_chw, blur_tf):
    
    image_complex_chw = tf.dtypes.complex(image_chw, tf.zeros_like(image_chw))
    image_fft_chw = tf.signal.fft2d(image_complex_chw)
    
    image_fft_mag_chw = mag_for_softmax(image_fft_chw, blur_tf)
    
    # and also we're subtracting fft_mag_max_chw, again for numerical stability.
    # It has no effect mathematically, but it keeps the values from overflowing.
    image_fft_mag_exp_chw = tf.exp(tf.cast((1.0 / softmax_temperature) * (image_fft_mag_chw - fft_mag_max_chw), dtype = tf.dtypes.float64))
    
    best_fft_mag_sum_chw.assign(best_fft_mag_sum_chw + image_fft_mag_exp_chw)
    best_fft_chw.assign(best_fft_chw + tf.cast(image_fft_chw, dtype = tf.dtypes.complex128) * tf.dtypes.complex(image_fft_mag_exp_chw, tf.zeros_like(image_fft_mag_exp_chw)))


image_count = 0
best_fft_mag_sum_chw = None
best_fft_chw = None
for filename in glob.glob(file_glob):
    for image_hwc, frame_index in ImageSequenceReader(filename, skip = light_frame_skip, to_float = True, demosaic = True):           
        image_count += 1
        
        if dark_image is not None:
            image_hwc = image_hwc - dark_image

        image_hwc = process_image(image_hwc, align_by_center_of_mass = align_by_center_of_mass, crop = crop)
        image_chw = hwc_to_chw(image_hwc)

        # we seem to need a crazy dynamic range, unless the temperature is really high, so use float64
        if best_fft_mag_sum_chw is None:
            best_fft_mag_sum_chw = tf.Variable(tf.zeros_like(image_chw, dtype = tf.dtypes.float64))
            best_fft_chw = tf.Variable(tf.zeros_like(image_chw, dtype = tf.dtypes.complex128))

        second_pass_step(best_fft_mag_sum_chw, best_fft_chw, image_chw, blur_tf)

        if light_frame_limit is not None and image_count >= light_frame_limit:
            break

best_fft_mag_sum_inv_chw = tf.math.reciprocal_no_nan(best_fft_mag_sum_chw)
best_fft_chw = best_fft_chw * tf.dtypes.complex(best_fft_mag_sum_inv_chw, tf.zeros_like(best_fft_mag_sum_inv_chw))

write_image(best_fft_chw, os.path.join(output_dir, f"fft_best.png"), frequency_domain = True)

best_chw = tf.math.abs(tf.signal.ifft2d(best_fft_chw))
best = chw_to_hwc(best_chw)
write_image(best, os.path.join(output_dir, f"best.png"), normalize = True)
write_image(center_image_per_channel(tf.cast(best, tf.float32)), os.path.join(output_dir, f"best_centered.png"), normalize = True)








