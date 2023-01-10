import tensorflow as tf

import os
import os.path
import tensorez.align as align
from tensorez.local_lucky import *
from tensorez.observation import *
from tensorez.util import *
from tensorez.image_sequence import *
from tensorez.align import *

import matplotlib.pyplot as plt

observation = Observation(
    lights = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'lights', '*.SER'), frame_step = 1),
    darks = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'darks', '*.SER')),
    align_by_content=True,
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
#   lights = ImageSequence(os.path.join('data', '2021-10-15_saturn_prime_crop', 'lights', '*.SER')),
#    darks = ImageSequence(os.path.join('data', '2021-10-15_saturn_prime_crop', 'darks', '*.SER')),
#    align_by_center_of_mass=True,
#    crop=(512, 512)
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


output_dir = create_timestamped_output_dir('pipeline')

debug_alignment = True
if debug_alignment and observation.align_by_content:
    observation.load_or_create_alignment_transforms()
    plt.figure(figsize=[11, 8.5])
    plt.plot(observation.alignment_transforms[:, 0], label = 'dx')
    plt.plot(observation.alignment_transforms[:, 1], label = 'dy')
    plt.plot(observation.alignment_transforms[:, 2], label = 'theta')
    if observation.alignment_transforms.shape[-1] > 3:
        plt.plot(observation.alignment_transforms[:, 3], label = 'log(sx)')
        plt.plot(observation.alignment_transforms[:, 4], label = 'log(sy)')
        plt.plot(observation.alignment_transforms[:, 5], label = 'skew')
    plt.xlabel('frame number')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'global_alignment.png'), dpi=600)
    plt.close()


#tf.config.run_functions_eagerly(True)
observation.debug_frame_limit = 1000

# more steps doesn't seem to make any difference
steps = 1
known_image = None
for step in range(steps):
    if steps > 1:
        step_output_dir = os.path.join(output_dir, f'step_{step}')
        os.makedirs(step_output_dir, exist_ok=True)    
    else:
        step_output_dir = output_dir

    known_image = local_lucky(
        observation,
        
        #algorithm = LuckinessAlgorithmLowpassAbsBandpass,
        #algorithm_kwargs = {'noise_wavelength_pixels': 15, 'crossover_wavelength_pixels': 45},

        algorithm=LuckinessAlgorithmFrequencyBands,
        algorithm_kwargs=dict(noise_wavelength_pixels=3, crossover_wavelength_pixels=20, isoplanatic_patch_pixels=50),

        #algorithm = LuckinessAlgorithmImageTimesKnown,
        #algorithm_kwargs=dict(isoplanatic_patch_pixels=50),

        #algorithm = LuckinessAlgorithmImageSquared,
        #algorithm_kwargs=dict(isoplanatic_patch_pixels=50),

        stdevs_above_mean = 2,
        steepness=3,
        debug_output_dir=step_output_dir,    
        bayer=True,
        drizzle=False,
        drizzle_kwargs={'upscale': 4, 'supersample': 4},
        average_image=known_image,
    )

    write_image(known_image, os.path.join(output_dir, f'step_{step}.png'))
