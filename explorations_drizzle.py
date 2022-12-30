import numpy as np
import tensorflow as tf
import os
from tensorez.util import *
from tensorez.image_sequence import *
from tensorez.align import *
from tensorez.drizzle import *

output_dir = create_timestamped_output_dir("drizzle")

test_image = read_image(os.path.join('data', 'synthetic', 'drizzle.png'))

alignment_transform = tf.constant([0.0, 0.0, 1.0, math.log(2), math.log(2), 0])
theta = alignment_transform_to_stn_theta(alignment_transform)

drizzled_image, drizzled_weights = drizzle(test_image, theta, upscale = 10, pixfrac = 0.5)

write_image(drizzled_image, os.path.join(output_dir, 'drizzled_image.png'))
write_image(drizzled_weights, os.path.join(output_dir, 'drizzled_weights.png'))
