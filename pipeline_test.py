import tensorflow as tf

import os
import os.path
import tensorez.align as align
from tensorez.local_lucky import *
from tensorez.observation import *
from tensorez.util import *
from tensorez.image_sequence import *
from tensorez.align import *

#observation = Observation(
#    lights = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'lights', '*.SER'), end_frame = 100),
#    darks = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'darks', '*.SER'))
#)

#observation = Observation(
#    lights = ImageSequence(os.path.join('data', '2021-10-15_saturn_prime_crop', 'lights', '*.SER')),
#    darks = ImageSequence(os.path.join('data', '2021-10-15_saturn_prime_crop', 'darks', '*.SER')),
#    align_by_center_of_mass=True,
#    crop=(512, 512)
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

observation = Observation(
    lights = ImageSequence(os.path.join('data', 'ISS_aligned_from_The_8_Bit_Zombie', '*.tif')),
    align_by_center_of_mass=True,
)




output_dir = create_timed_output_dir('pipeline')

if True:
    alignment_transforms = compute_alignment_transforms(
        observation,
        debug_output_dir=output_dir,
        max_steps = 100
    )
    observation.set_alignment_transforms(alignment_transforms)

final_image = local_lucky(
    observation,
    stdevs_above_mean = 3,
    debug_output_dir=output_dir,
)
write_image(final_image, os.path.join(output_dir, 'final_image.png'))
