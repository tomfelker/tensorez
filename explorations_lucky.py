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
frame_limit = 99999

###############################################################################
# Data selection

#file_glob = os.path.join('data', '2021-10-15_jupiter_prime', '2021-10-16-0439_9-CapObj.SER'); file_glob_darks = os.path.join('data', '2021-10-15_jupiter_prime', '2021-10-16-0441_0-CapObj.SER');
#file_glob = os.path.join('data', '2021-10-15_jupiter_barlow3x', '2021-10-16-0501_0-CapObj.SER'); file_glob_darks = os.path.join('data', '2021-10-15_jupiter_barlow3x', '2021-10-16-0506_5-CapObj.SER');
file_glob = os.path.join('data', '2021-10-15_saturn_prime_crop', '2021-10-16-0445_8-CapObj.SER'); file_glob_darks = os.path.join('data', '2021-10-15_saturn_prime_crop', '2021-10-16-0448_0-CapObj.SER');
#file_glob = os.path.join('data', '2021-10-15_jupiter_prime', 'fake_jupiter.png');

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

def process_image(image_hwc, align_by_center_of_mass, crop):
    # todo: this is dumb, replace with real alignment
    # todo: it also probably works best if images are mean 0, could do that...
    if align_by_center_of_mass:
        align_steps = 0
        while True:                
            image_hwc, shift, shift_axis = center_image(image_hwc, only_even_shifts = only_even_shifts)
            align_steps += 1
            print(f"Shifted by {shift}")
            if tf.reduce_max(tf.abs(shift)) < (2 if only_even_shifts else 1):
                print(f"Centered after {align_steps} steps.")
                break
            if align_steps >= max_align_steps:
                print(f"Alignment didn't converge after {align_steps} steps.")
                break
    
    if crop is not None:
        image_hwc = crop_image(image_hwc, crop, crop_align = 1)

    return image_hwc        

def load_average_image(file_glob, frame_limit = None, dark_image_to_subtract = None, **process_image_args):
    average_image = None
    image_count = 0
    for filename in glob.glob(file_glob):
        for image_hwc, frame_index in ImageSequenceReader(filename, to_float = True, demosaic = True):           

            if dark_image_to_subtract is not None:
                image_hwc = image_hwc - dark_image_to_subtract

            image_hwc = process_image(image_hwc, **process_image_args)

            if average_image is None:
                average_image = image_hwc
            else:
                average_image += image_hwc

            image_count += 1
            if frame_limit is not None and image_count >= frame_limit:
                break
    if image_count > 0:
        average_image *= 1 / image_count
    return average_image

dark_image = None
if file_glob_darks is not None:
    dark_image = load_average_image(file_glob_darks, crop = None, frame_limit = frame_limit, align_by_center_of_mass = False)
    write_image(dark_image, os.path.join(output_dir, 'dark.png'))
    write_image(dark_image, os.path.join(output_dir, 'dark_normalized.png'), normalize = True)

average_image = load_average_image(file_glob, dark_image_to_subtract = dark_image, crop = crop, frame_limit = frame_limit, align_by_center_of_mass = align_by_center_of_mass)
write_image(average_image, os.path.join(output_dir, 'average.png'))

average_image_chw = hwc_to_chw(average_image)
average_fft_complex_chw = tf.dtypes.complex(average_image_chw, tf.zeros_like(average_image_chw))
average_fft_chw = tf.signal.fftshift(tf.signal.fft2d(average_fft_complex_chw))
average_fft_mag_chw = tf.abs(average_fft_chw)

average_fft_mag_chw = tf.math.log(average_fft_mag_chw)

average_fft_mag = chw_to_hwc(average_fft_mag_chw)
write_image(average_fft_mag, os.path.join(output_dir, 'average_fft_mag.png'))
write_image(average_fft_mag, os.path.join(output_dir, 'average_fft_mag_normalized.png'), normalize = True)

image_index = 0
fft_max_mag_chw = None
fft_max_chw = None
for filename in glob.glob(file_glob):
    for image_hwc, frame_index in ImageSequenceReader(filename, to_float = True, demosaic = True):           
        image_index += 1

        image_hwc = process_image(image_hwc, align_by_center_of_mass = align_by_center_of_mass, crop = crop)

        if dark_image is not None:
                image_hwc = image_hwc - dark_image

        image_chw = hwc_to_chw(image_hwc)
        fft_complex_chw = tf.dtypes.complex(image_chw, tf.zeros_like(image_chw))
        fft_chw = tf.signal.fft2d(fft_complex_chw)
        
        fft_mag_chw = tf.abs(fft_chw)

        if fft_max_mag_chw is None:
            fft_max_mag_chw = fft_mag_chw
            fft_max_chw = fft_chw
        else:
            fft_max_mag_chw = tf.maximum(fft_max_mag_chw, fft_mag_chw)

        fft_best_mask_chw = tf.equal(fft_max_mag_chw, fft_mag_chw)            
        fft_max_chw = tf.where(fft_best_mask_chw, fft_chw, fft_max_chw)

        if False:            
            fft_log_mag_chw = tf.math.log(tf.signal.fftshift(fft_mag_chw))
            fft_log_mag = chw_to_hwc(fft_log_mag_chw)
            write_image(fft_log_mag, os.path.join(output_dir, f"fft_log_mag_normalized_{image_index:08d}.png"), normalize = True)

            fft_phase_chw = tf.signal.fftshift(tf.atan2(tf.math.real(fft_chw), tf.math.imag(fft_chw)))
            fft_phase = chw_to_hwc(fft_phase_chw)
            write_image(fft_phase, os.path.join(output_dir, f"fft_phase_normalized_{image_index:08d}.png"), normalize = True)
        
        if frame_limit is not None and image_index >= frame_limit:
            break

    if True:
        fft_log_max_mag_chw = tf.math.log(tf.signal.fftshift(fft_max_mag_chw))
        fft_log_max_mag = chw_to_hwc(fft_log_max_mag_chw)
        write_image(fft_log_max_mag, os.path.join(output_dir, f"fft_log_max_mag_normalized.png"), normalize = True)

        fft_max_phase_chw = tf.signal.fftshift(tf.atan2(tf.math.real(fft_chw), tf.math.imag(fft_chw)))
        fft_max_phase = chw_to_hwc(fft_max_phase_chw)
        write_image(fft_max_phase, os.path.join(output_dir, f"fft_max_phase_normalized.png"), normalize = True)

        best_chw = tf.math.abs(tf.signal.ifft2d(fft_max_chw))
        best = chw_to_hwc(best_chw)
        write_image(best, os.path.join(output_dir, f"best.png"), normalize = True)








