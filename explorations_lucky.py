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
from tensorez.image_sequence import *
from tensorez.model import *
from tensorez.bayer import *
from tensorez.fourier import *
from tensorez.obd import *

###############################################################################
# Settings
align_by_center_of_mass = True
crop = None

only_even_shifts = False
max_align_steps = 50

max_freq = 1 / 5
min_freq = 1 / 50

quantiles = [0, .01, .05, .1, .3, .5, .9, 1]

###############################################################################
# Data selection

#file_glob = os.path.join('data', '2021-10-15_jupiter_prime', 'jupiter*.light.SER'); file_glob_darks = os.path.join('data', '2021-10-15_jupiter_prime', 'jupiter*.dark.SER'); crop = (2048, 2048)
#file_glob = os.path.join('data', '2021-10-15_jupiter_prime_crop', '*.light.SER'); file_glob_darks = os.path.join('data', '2021-10-15_jupiter_prime_crop', '*.dark.SER'); crop = (512, 512)
#file_glob = os.path.join('data', '2021-10-15_jupiter_barlow3x', '2021-10-16-0501_0-CapObj.SER'); file_glob_darks = os.path.join('data', '2021-10-15_jupiter_barlow3x', '2021-10-16-0506_5-CapObj.SER');
#file_glob = os.path.join('data', '2021-10-15_saturn_prime_crop', '2021-10-16-0445_8-CapObj.SER'); file_glob_darks = os.path.join('data', '2021-10-15_saturn_prime_crop', '2021-10-16-0448_0-CapObj.SER');
#file_glob = os.path.join('data', '2021-10-15_jupiter_prime', 'fake_jupiter.png');
#file_glob = os.path.join('data', 'ISS_aligned_from_The_8_Bit_Zombie', '*.tif')
#file_glob = os.path.join('data', '2020-11-21Z_iss', '2020-11-21-0222_4-CapObj.SER'); light_frame_limit = 1000; light_frame_skip = 2900

#file_glob = os.path.join('data', '2022-12-07_moon_mars_conjunction', '2022-12-08-0742_7-CapObj.SER'); #crop = (512, 512); #file_glob_darks = os.path.join('data', '2022-12-07_moon_mars_conjunction', '2022-12-08-0746_0-CapObj.SER')

# Mars, directly imaged, near opposition, somewhat dewy lens
lights = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_prime', 'lights', '*.SER'), frame_step = 1)
darks = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_prime', 'darks', '*.SER'), frame_step = 1)
crop = (512, 512)

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
    
def compute_ratings(lights, dark_image):
    fft_weights_chw = None
    ratings = []
    for image_index, image_hwc in enumerate(lights):

        if dark_image is not None:
            image_hwc -= dark_image
        image_hwc = process_image(image_hwc, align_by_center_of_mass = align_by_center_of_mass, crop = crop)
        image_chw = hwc_to_chw(image_hwc)

        if fft_weights_chw is None:
            fft_weights_chw = get_fft_weights_chw(image_chw.shape)
            write_image(chw_to_hwc(fft_weights_chw), os.path.join(output_dir, 'fft_weights.png'), normalize = True)

        rating = rate_image(image_chw, fft_weights_chw)
        ratings.append(rating.numpy()[0])

        print(f"Image {image_index + 1} of {len(lights)} has rating {rating}")
    return ratings


dark_image = None
if darks is not None:
    dark_image = darks.read_average_image()
    write_image(dark_image, os.path.join(output_dir, 'dark.png'))
    write_image(dark_image, os.path.join(output_dir, 'dark_normalized.png'), normalize = True)

ratings = compute_ratings(lights, dark_image)

rating_and_index_pairs = []
for index, rating in enumerate(ratings):
    rating_and_index_pairs.append((rating, index))

rating_and_index_pairs.sort(reverse = True)

image_sum = None
image_count = 0
quantile_index = 0
for rank, (rating, image_index) in enumerate(rating_and_index_pairs):
    write_file = False    
    quantile = quantiles[quantile_index]
    if (rank / len(ratings) >= quantile) or (rank == len(ratings) - 1):
        write_file = True
        quantile_index += 1

    print(f"Loading the {rank + 1}th best image, rating {rating}, index {image_index}")
    
    image_hwc = lights.read_image(image_index)
    if dark_image is not None:
        image_hwc -= dark_image
    image_hwc = process_image(image_hwc, align_by_center_of_mass = align_by_center_of_mass, crop = crop)
    image_count += 1

    if image_sum is None:
        image_sum = tf.Variable(tf.zeros_like(image_hwc))

    image_sum.assign(image_sum + image_hwc)

    if write_file:
        image_average = image_sum * (1.0 / image_count)
        
        if quantile == 1:            
            quantile_str = 'mean'
        elif quantile == 0:
            quantile_str = 'best_frame'
        else:
            quantile_str = f"best_{quantile:f}".replace('0.', 'point_')
        filename = f"{quantile_str}_including_{image_count}_of_{len(ratings)}.png"

        write_image(image_average, os.path.join(output_dir, filename))

    if quantile_index >= len(quantiles):
        break

