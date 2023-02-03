import os

#os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
import tensorflow as tf


import os.path
from tensorez.observation import *
from tensorez.util import *
from tensorez.image_sequence import *
from tensorez.local_align import *

import matplotlib.pyplot as plt

observation = Observation(
    lights = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'lights', '*.SER'), frame_step = 1),
    darks = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'darks', '*.SER')),
    #align_by_content=True,
    #compute_alignment_transforms_kwargs={'allow_scale': True, 'allow_skew': True}
)

#observation = Observation(
#    lights = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_prime', 'lights', '*.SER')),
#    darks = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'mars_prime', 'darks', '*.SER')),
#    align_by_center_of_mass=True,
#    align_by_center_of_mass_only_even_shifts=True,
#    crop=(512, 512),
#    crop_align=2,
#    crop_before_content_align=True,
#    align_by_content=True,
#    compute_alignment_transforms_kwargs={'allow_rotation': False, 'allow_scale': False, 'allow_skew': False}
#)

#observation = Observation(
#    lights = ImageSequence(os.path.join('data', '2021-10-15_saturn_prime_crop', 'lights', '*.SER')),
#    darks = ImageSequence(os.path.join('data', '2021-10-15_saturn_prime_crop', 'darks', '*.SER')),
#  
#    align_by_center_of_mass=True,
#    align_by_center_of_mass_only_even_shifts=True,
#    crop=(512, 512),
#    crop_align=2,
#    crop_before_content_align=True,
#    align_by_content=True,
#    compute_alignment_transforms_kwargs={'allow_rotation': False, 'allow_scale': False, 'allow_skew': False}
#)

#observation = Observation(
#   lights = ImageSequence(os.path.join('data', '2021-10-15_jupiter_prime', 'jupiter_1.light.SER')),
#    darks = ImageSequence(os.path.join('data', '2021-10-15_jupiter_prime', 'jupiter.dark.SER')),
#    align_by_center_of_mass=True,
#    crop=(2048, 2048)
#)

#observation = Observation(
#    lights = ImageSequence(os.path.join('data', 'saturn_bright_mvi_6902', '0*.png')),
#    align_by_center_of_mass=True,
#)

#observation = Observation(
#    lights = ImageSequence(os.path.join('data', 'ser_player_examples', 'Jup_200415_204534_R_F0001-0300.ser')),
#    align_by_center_of_mass=True,
#)

#observation = Observation(
#    lights = ImageSequence(os.path.join('data', 'ser_player_examples', 'Mars_150414_002445_OSC_F0001-0500.ser')),
#    align_by_center_of_mass=True,
#)

#observation = Observation(
#    lights = ImageSequence(os.path.join('data', 'ISS_aligned_from_The_8_Bit_Zombie', '*.tif')),
#    align_by_center_of_mass=True,
#    align_by_content=True,
#    compute_alignment_transforms_kwargs={'allow_scale': True, 'allow_skew': True}
#)


output_dir = create_timestamped_output_dir('local_align')


#tf.config.run_functions_eagerly(True)
observation.debug_frame_limit = 100

local_align(
    lights=observation,
    alignment_output_dir = output_dir, # todo: caching
    debug_output_dir = output_dir,
    max_steps=500
)

