"""
Lucky Imaging

Make one pass to align and center everything, one pass to score everything, and add up accordign to scores

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
debug_frame_limit = 5

max_freq = 1 / 5
min_freq = 1 / 50

quantiles = [0.00000001, .01, .05, .1, .3, .5]

###############################################################################
# Data selection

file_glob = os.path.join('data', '2021-10-15_jupiter_prime', 'jupiter*.light.SER'); file_glob_darks = os.path.join('data', '2021-10-15_jupiter_prime', 'jupiter*.dark.SER'); crop = (2048, 2048)
#file_glob = os.path.join('data', '2021-10-15_jupiter_prime_crop', '*.light.SER'); file_glob_darks = os.path.join('data', '2021-10-15_jupiter_prime_crop', '*.dark.SER'); crop = (512, 512)
#file_glob = os.path.join('data', '2021-10-15_jupiter_barlow3x', '2021-10-16-0501_0-CapObj.SER'); file_glob_darks = os.path.join('data', '2021-10-15_jupiter_barlow3x', '2021-10-16-0506_5-CapObj.SER');
#file_glob = os.path.join('data', '2021-10-15_saturn_prime_crop', '2021-10-16-0445_8-CapObj.SER'); file_glob_darks = os.path.join('data', '2021-10-15_saturn_prime_crop', '2021-10-16-0448_0-CapObj.SER');
#file_glob = os.path.join('data', '2021-10-15_jupiter_prime', 'fake_jupiter.png');
#file_glob = os.path.join('data', 'ISS_aligned_from_The_8_Bit_Zombie', '*.tif')
#file_glob = os.path.join('data', '2020-11-21Z_iss', '2020-11-21-0222_4-CapObj.SER'); light_frame_limit = 1000; light_frame_skip = 2900

#file_glob = os.path.join('data', '2022-12-07_moon_mars_conjunction', '2022-12-08-0742_7-CapObj.SER'); #crop = (512, 512); #file_glob_darks = os.path.join('data', '2022-12-07_moon_mars_conjunction', '2022-12-08-0746_0-CapObj.SER')

#file_glob = os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_prime', 'lights', '*.SER'); crop = (512, 512); file_glob_darks = os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_prime', 'darks', '*.SER')
#file_glob = os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_3x', 'lights', '*.SER'); crop = (1024, 1024); file_glob_darks = os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_3x', 'darks', '*.SER')

#file_glob = os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_prime', 'stars', '*.SER'); crop = (1024, 1024); light_frame_limit = 1328; #file_glob_darks = os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_prime', 'darks', '*.SER')
#file_glob = os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_3x', 'stars', '*.SER'); crop = (1024, 1024); #file_glob_darks = os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_3x', 'darks', '*.SER')


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

@tf.function
def rate_image(image_chw, fft_weights_chw):
    image_chw = tf.dtypes.complex(image_chw, tf.zeros_like(image_chw))
    image_fft_chw = tf.signal.fft2d(image_chw)
    image_fft_mag_chw = tf.abs(image_fft_chw)
    # hmm, also reducing across channels - but maybe we don't want that, as channels may be independently in focus?
    return tf.reduce_mean(tf.multiply(image_fft_mag_chw, fft_weights_chw), axis = [-1, -2, -3])

def get_fft_weights_chw(shape):
    w = shape[-1]
    h = shape[-2]
    freqs_x = tf.convert_to_tensor(np.fft.fftfreq(w))
    freqs_y = tf.convert_to_tensor(np.fft.fftfreq(h))
    freqs_x = tf.expand_dims(freqs_x, axis = [-2])
    freqs_y = tf.expand_dims(freqs_y, axis = [-1])
    freqs = tf.sqrt(freqs_x * freqs_x + freqs_y * freqs_y)
    lowpass = tf.greater(freqs, min_freq)
    highpass = tf.less(freqs, max_freq)
    bandpass = tf.cast(tf.logical_and(highpass, lowpass), dtype = tf.float32)
    return tf.expand_dims(bandpass, axis = [-3])
    

dark_image = None
if file_glob_darks is not None:
    dark_image, image_count = load_average_image(file_glob_darks, frame_limit = dark_frame_limit)
    write_image(dark_image, os.path.join(output_dir, 'dark.png'))
    write_image(dark_image, os.path.join(output_dir, 'dark_normalized.png'), normalize = True)

image_count = 0
average_chw = None
fft_weights_chw = None
ratings = []
for filename in glob.glob(file_glob):
    for image_hwc, frame_index in ImageSequenceReader(filename, skip = light_frame_skip, to_float = True, demosaic = True):           
        image_count += 1

        if light_frame_limit is not None and image_count > light_frame_limit:
            break

        if dark_image is not None:
            image_hwc -= dark_image

        image_hwc = process_image(image_hwc, align_by_center_of_mass = align_by_center_of_mass, crop = crop)

        image_chw = hwc_to_chw(image_hwc)

        if average_chw is None:
            average_chw = tf.Variable(tf.zeros_like(image_chw))

        if fft_weights_chw is None:
            fft_weights_chw = get_fft_weights_chw(image_chw.shape)
            write_image(chw_to_hwc(fft_weights_chw), os.path.join(output_dir, 'fft_weights.png'), normalize = True)

        average_chw.assign(average_chw + image_chw)

        rating = rate_image(image_chw, fft_weights_chw)
        ratings.append(rating.numpy()[0])

        print(f"finished pass 1 step {image_count}, rating was {rating}")


if image_count == 0:
    raise RuntimeError(f"Couldn't load any images from '{file_glob}'.")      

average_chw.assign(average_chw * (1.0 / image_count))

write_image(chw_to_hwc(average_chw), os.path.join(output_dir, 'average.png'), normalize = True)

#todo: if we had random access to images, we could spit out tons of quantiles in one pass / with only one sum...
# but for now, just do several discrete steps, with more memory cost.

sorted_ratings = sorted(ratings, reverse = True)

quantile_rating_cutoffs = []
for quantile in quantiles:
    quantile_rating_cutoffs.append(sorted_ratings[int(quantile * len(ratings))])

image_count = 0
quantile_images_hwc = None
quantile_counts = None
for filename in glob.glob(file_glob):
    for image_hwc, frame_index in ImageSequenceReader(filename, skip = light_frame_skip, to_float = True, demosaic = True):           
        image_count += 1
        image_index = image_count - 1

        if light_frame_limit is not None and image_count > light_frame_limit:
            break

        need_processing = False

        for quantile_index, quantile in enumerate(quantiles):
            if ratings[image_index] >= quantile_rating_cutoffs[quantile_index]:
                need_processing = True
                break

        if not need_processing:
            continue

        if dark_image is not None:
            image_hwc -= dark_image
        
        image_hwc = process_image(image_hwc, align_by_center_of_mass = align_by_center_of_mass, crop = crop)

        if quantile_images_hwc is None:
            quantile_images_hwc = []
            quantile_counts = []
            for quantile in quantiles:
                quantile_images_hwc.append(tf.Variable(tf.zeros_like(image_hwc)))
                quantile_counts.append(0)

        for quantile_index, quantile in enumerate(quantiles):
            if ratings[image_index] >= quantile_rating_cutoffs[quantile_index]:
                quantile_images_hwc[quantile_index].assign(quantile_images_hwc[quantile_index] + image_hwc)
                quantile_counts[quantile_index] += 1
                
        print(f"finished pass 2 step {image_count}")


for quantile_index, quantile in enumerate(quantiles):
    quantile_images_hwc[quantile_index].assign(quantile_images_hwc[quantile_index] * (1.0 / quantile_counts[quantile_index]))

for quantile_index, quantile in enumerate(quantiles):
    quantile_percent_str = str(quantile * 100).replace('.', '_point_')
    filename = f"top_{quantile_percent_str}_percent_{quantile_counts[quantile_index]}_of_{len(ratings)}.png"
    write_image(quantile_images_hwc[quantile_index], os.path.join(output_dir, filename))