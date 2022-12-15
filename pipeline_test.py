import tensorflow as tf

import os
import os.path
import tensorez.align as align
from tensorez.local_lucky import *
from tensorez.observation import *
from tensorez.util import *
from tensorez.image_sequence import *
from tensorez.align import *

observation = Observation(
    lights = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'lights', '*.SER')),
    darks = ImageSequence(os.path.join('data', '2022-12-07_moon_mars_conjunction', 'moon_prime', 'darks', '*.SER'))
)


output_dir = create_timed_output_dir('pipeline')

alignment_transforms = compute_alignment_transforms(
    observation,
    debug_output_dir=output_dir,
    max_steps = 10
)
observation.set_alignment_transforms(alignment_transforms)

final_image = local_lucky(observation, debug_output_dir=output_dir)
write_image(final_image, os.path.join(output_dir, 'final_image.png'))
