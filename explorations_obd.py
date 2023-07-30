import numpy as np
import tensorflow as tf
import os
import glob
import datetime
import gc
import sys
import time
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
initial_blur_stddev_px = 0
dark_ratio = 1

frame_skip = 0
frame_limit = None
dark_frame_limit = None
max_align_steps = 50
only_even_shifts = False
# (power of 2) + 1 is most efficient
psf_size = 256+1


###############################################################################
# Data selection

#file_glob = os.path.join('data', 'firecapture_examples', 'jup', 'Jup_fixedbayer.ser')
#file_glob = os.path.join('data', 'jupiter_mvi_6906', '????????.png')
#file_glob = os.path.join('data', 'ISS_aligned_from_The_8_Bit_Zombie', '*.tif')
#file_glob = os.path.join('data', 'saturn_bright_mvi_6902', '????????.png')
#file_glob = os.path.join('data', 'powerline_t4i_raw', '*.cr2'); align_by_center_of_mass = False
#file_glob = os.path.join('data', 'ASICAP', 'CapObj', '2020-07-30Z', '2020-07-30-2020_6-CapObj.SER'); align_by_center_of_mass = False;
#file_glob = os.path.join('data', 'ASICAP', 'CapObj', '2020-11-21Z','2020-11-21-0222_4-CapObj.SER'); align_by_center_of_mass = True; frame_skip = 3500; frame_limit = 475; crop = (768, 768)

#file_glob = os.path.join('data', 'ASICAP', 'CapObj', '2020-12-21Z','2020-12-21-0331_1-CapObj.SER'); align_by_center_of_mass = True; crop = (768, 768)
#file_glob = os.path.join('data', 'ASICAP', 'CapObj', '2020-12-21Z','2020-12-21-0332_0-CapObj.SER'); align_by_center_of_mass = True; crop = (768, 768)

#best one?
#file_glob = os.path.join('data', 'ASICAP', 'CapObj', '2020-12-21Z','2020-12-21-0332_6-CapObj.SER'); align_by_center_of_mass = True; crop = (768, 768)

#file_glob = os.path.join('data', 'ASICAP', 'CapObj', '2020-12-21Z','2020-12-21-0333_3-CapObj.SER'); align_by_center_of_mass = True; crop = (768, 768)

# brighter, but dunno if it corresponds to the dark...
#file_glob = os.path.join('data', 'great_conjunction_MVI_7637', '????????.png'); align_by_center_of_mass = False; crop = None


#file_glob = os.path.join('data', 'great_conjunction_MVI_7648', '????????.png'); align_by_center_of_mass = True; crop = None; file_glob_darks = os.path.join('data', 'great_conjunction_darks_MVI_7649', '????????.png')

#file_glob = os.path.join('data', 'ASICAP', 'CapObj', '2020-12-22Z','2020-12-22-0148_3-CapObj.SER'); align_by_center_of_mass = False; file_glob_darks = os.path.join('data', 'ASICAP', 'CapObj', '2020-12-22Z','2020-12-22-0209_3-CapObj.SER'); dark_ratio = 510/330

#dark ratio 6 * 510/330 = 9.27 was too much, 5 * 510/330  = 7.72 not enough

#file_glob = os.path.join('data', 'ASICAP', 'CapObj', '2020-12-22Z','2020-12-22-0148_3-CapObj.SER'); align_by_center_of_mass = False; initial_blur_stddev_px = 0; file_glob_darks = os.path.join('data', 'ASICAP', 'CapObj', '2020-12-30Z','2020-12-30-0303_5-CapObj.SER'); dark_ratio = 1; frame_limit = 100; dark_frame_limit = 100; crop = (3072, 2048)

#file_glob = os.path.join('data', 'ASICAP', 'CapObj', '2020-12-22Z','2020-12-22-0148_3-CapObj.SER'); align_by_center_of_mass = False; initial_blur_stddev_px = 0; frame_limit = 100; crop = (3072, 2048)

#file_glob = os.path.join('data', 'ASICAP', 'CapObj', '2020-12-22Z','2020-12-22-0148_3-CapObj.SER'); align_by_center_of_mass = False; initial_blur_stddev_px = 0; frame_limit = 100; crop = (384, 384); crop_center =(492, 1468)


#file_glob = os.path.join('data', '2021-10-15_jupiter_prime_crop', '2021-10-16-0428_1-CapObj.SER'); crop = (1024, 1024); file_glob_darks = os.path.join('data', '2021-10-15_jupiter_prime_crop', '2021-10-16-0430_9-CapObj.SER');

#file_glob = os.path.join('data', '2021-10-15_jupiter_prime', '2021-10-16-0439_9-CapObj.SER'); file_glob_darks = os.path.join('data', '2021-10-15_jupiter_prime', '2021-10-16-0441_0-CapObj.SER');

file_glob = os.path.join('data', '2022-12-07_moon_mars_conjunction', '2022-12-08-0742_7-CapObj.SER'); crop = (512, 512)

###############################################################################
# Code

output_dir = os.path.join("output", "latest_obd", datetime.datetime.now().replace(microsecond = 0).isoformat().replace(':', '_'))

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

def load_and_process_average_image(file_glob, frame_limit = None, frame_skip = None, dark_image_to_subtract = None, **process_image_args):
    average_image = None
    image_count = 0
    for filename in glob.glob(file_glob):
        for image_hwc, frame_index in ImageSequenceReader(filename, skip = frame_skip, to_float = True, demosaic = True):           

            image_hwc = process_image(image_hwc, **process_image_args)

            if dark_image_to_subtract is not None:
                image_hwc = image_hwc - dark_image_to_subtract

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
    dark_image = load_and_process_average_image(file_glob_darks, frame_limit = dark_frame_limit, crop = crop, align_by_center_of_mass = False)
    write_image(dark_image, os.path.join(output_dir, 'dark.png'))
    write_image(dark_image, os.path.join(output_dir, 'dark_normalized.png'), normalize = True)

estimated_image = load_and_process_average_image(file_glob, frame_limit = frame_limit, frame_skip = frame_skip, dark_image_to_subtract = dark_image, crop = crop, align_by_center_of_mass = align_by_center_of_mass)
write_image(estimated_image, os.path.join(output_dir, 'initial_estimated_image.png'))
write_image(estimated_image, os.path.join(output_dir, 'initial_estimated_image_normalized.png'), normalize = True)

#bias = -tf.reduce_min(estimated_image, keep_dims = True) + .2
bias = .2
print(f'bias: {bias}')
estimated_image += bias

estimated_image = hwc_to_chw(pad_for_conv(estimated_image, psf_size))

if initial_blur_stddev_px != 0:
    initial_blur_psf = gaussian_psf(psf_size, initial_blur_stddev_px / psf_size)
    initial_blur_psf = tf.expand_dims(initial_blur_psf, axis = -4)
    write_image(initial_blur_psf, os.path.join(output_dir, 'initial_blur_psf.png'), normalize = True)
    
    initial_blur_psf = hwc_to_chw(initial_blur_psf)    

    estimated_image = conv2d_fourier(estimated_image, initial_blur_psf)

    # switch back for writing, and for pad_for_conv
    estimated_image = chw_to_hwc(estimated_image)

    write_image(estimated_image, os.path.join(output_dir, 'average_image_blurred.png'))

    estimated_image = pad_for_conv(estimated_image, psf_size)
    estimated_image = hwc_to_chw(estimated_image)


psf_shape = (1, 3, psf_size, psf_size)
step = 0
epoch = 0
while True:
    step_in_epoch = 0
    image_count = 0
    loop_start_time = time.perf_counter()
    for filename in glob.glob(file_glob):
        for image, frame_index in ImageSequenceReader(filename, skip = frame_skip, to_float = True, demosaic = True):
            
            image = process_image(image, align_by_center_of_mass = align_by_center_of_mass, crop = crop)
            
            if dark_image is not None:
                image = image - dark_image

            image += bias

            image = hwc_to_chw(image)
            
            write_sequential_image(chw_to_hwc(image), output_dir, 'image', step, normalize = False, saturate = False)
            write_sequential_image(chw_to_hwc(estimated_image), output_dir, 'estimated_image', step, normalize = False, saturate = False)

            estimated_image, estimated_psf = obd_step(estimated_image, image, psf_shape)    

            write_every = 1 << epoch
            if step_in_epoch % write_every == 0:
                

                estimated_image_debiased = estimated_image - bias                
                write_sequential_image(chw_to_hwc(estimated_image_debiased), output_dir, 'estimated_image_debiased', step, normalize = False, saturate = False)
                write_sequential_image(chw_to_hwc(estimated_image_debiased), output_dir, 'estimated_image_debiased_normalized', step, normalize = True, saturate = False)
                write_sequential_image(chw_to_hwc(estimated_image_debiased), output_dir, 'estimated_image_debiased_normalized_saturated', step, normalize = False, saturate = True)
                del estimated_image_debiased
                write_sequential_image(chw_to_hwc(estimated_psf), output_dir, 'estimated_psf', step, normalize = True)
            
            step += 1
            step_in_epoch += 1
            perf_counter = time.perf_counter()
            print(f'loop time: {perf_counter - loop_start_time}')
            loop_start_time = perf_counter

            image_count += 1
            if frame_limit is not None and image_count >= frame_limit:
                break
    epoch += 1
