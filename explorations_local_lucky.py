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


from threading import currentThread
from tkinter import N
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
from tensorez.local_lucky import *
from tensorez.observation import *

###############################################################################
# Settings
align_by_center_of_mass = True
crop = None
darks = None

only_even_shifts = False
max_align_steps = 50

debug_frames = 2

###############################################################################
# Data selection

# Moon, directly imaged, near opposition, somewhat dewy lens
#lights = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'lights', '*.SER'), frame_step=1, end_frame=30)
#darks = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'darks', '*.SER'), frame_step=1, end_frame=30)
#align_by_center_of_mass = False

# Mars, directly imaged, near opposition, somewhat dewy lens
#lights = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_prime', 'lights', '*.SER'), frame_step=1, end_frame=None)
#darks = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_prime', 'darks', '*.SER'), frame_step=1, end_frame=None)
#align_by_center_of_mass = True
#crop = (512, 512)

# Mars, barlow, near opposition, somewhat dewy lens
#lights = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_3x', 'lights', '*.SER'), frame_step=1, end_frame=None)
#darks = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_3x', 'darks', '*.SER'), frame_step=1, end_frame=None)
#align_by_center_of_mass = True
#crop = (1024, 1024)


# Jupiter
#lights = ImageSequence(os.path.join('data', '2021-10-15_jupiter_prime', 'jupiter*.light.SER'), end_frame = None)
#darks = ImageSequence(os.path.join('data', '2021-10-15_jupiter_prime', 'jupiter*.dark.SER'), end_frame = None)
#align_by_center_of_mass = True
#crop = (2048, 2048)

# ISS
observation = Observation(
    lights = ImageSequence(os.path.join('data', '2023-07-30_iss', '*.SER'), start_frame=155, end_frame=205),
    align_by_center_of_mass=True
)

crop=(512,512)

###############################################################################
# Functions


###############################################################################
# Code

local_lucky(observation, LuckinessAlgorithmFrequencyBands, algorithm_kwargs={'isoplanatic_patch_pixels': 50, 'crossover_wavelength_pixels': 5,  'noise_wavelength_pixels': 2})

