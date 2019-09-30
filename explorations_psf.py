import numpy as np
import tensorflow as tf
import os
import glob
import datetime
import math
from tensorez.model import *
from tensorez.util import *

tf.compat.v1.enable_eager_execution()

output_dir = os.path.join('output', 'explorations', 'psf')

#before getting too deep, let's just init these more reasonably:


write_image(gaussian_psf(64), os.path.join(output_dir, "psf_guess.png"), normalize = True)


# all the below is cool, but probably not useful



# pretend telescope stats
aperture_diameter_m = 280e-3
obstruction_diameter_m = 95e-3
focal_length_m = 2800e-3
barlow_factor = 1
pixel_size_m = 4.3e-6
wavelengths_m = [700e-9, 530e-9, 470e-9]


effective_focal_length_m = focal_length_m * barlow_factor



aperture_function_size = 6400

#hmm, I bet there's a principled way to figure this...
zoom_factor = 100


#Kolmogorov something something
# no_noise_below r_0 something something
def turbulence_2d(size, no_noise_at_scale = 3):
    turbulence = tf.random.normal(shape = (1, 1, 1, 1))
    steps = math.ceil(math.log(size, 2))
    for step in range(0, steps):
        # why is this hard? just want to upsample
        print("resizing to {}".format(turbulence.shape[-2]))
        turbulence = tf.image.resize(turbulence, size = (turbulence.shape[-2] * 2, turbulence.shape[-3] * 2), method = tf.image.ResizeMethod.BICUBIC)
        if steps - step - 1 >= no_noise_at_scale:
            turbulence += tf.random.normal(shape = turbulence.shape)
    turbulence = tf.squeeze(turbulence, axis = -4)
    turbulence = tf.squeeze(turbulence, axis = -1)
    return turbulence[0:size, 0:size]
        
write_image(turbulence_2d(512), os.path.join(output_dir, "turbulence_test.png"), normalize = True)


def complex_to_unit_norm(complex_tensor, tolerance = 1e-9):
    divisor = tf.abs(complex_tensor)
    divisor = tf.maximum(divisor, tolerance)
    return complex_tensor / tf.dtypes.complex(divisor, divisor)


# todo: compute smaller wavefront to match aperture size, pad appropriately
#wavefront_size = math.ceil(aperture_function_size / zoom_factor)
wavefront_size = aperture_function_size

# todo, don't think this is right - shouldn't it be more like, turbulence chooses temperature chooses optical path length, over wavelength mod tau gives phase?
# but it's hand-wavingly right?  my arms are tired.
wavefront = tf.dtypes.complex(turbulence_2d(wavefront_size), turbulence_2d(wavefront_size))
wavefront = complex_to_unit_norm(wavefront)

def complex_to_image(complex_tensor):
    return tf.stack([tf.math.real(complex_tensor) + .5, tf.math.imag(complex_tensor) + .5, tf.zeros(complex_tensor.shape)], axis = -1)

#write_image(complex_to_image(wavefront), os.path.join(output_dir, "wavefront.png"))

aperture_softness_px = 1
aperture_softness_m = aperture_diameter_m / aperture_function_size * aperture_softness_px * zoom_factor

aperture_function_coord = tf.linspace(-zoom_factor * aperture_diameter_m / 2.0, zoom_factor * aperture_diameter_m / 2.0, num = aperture_function_size)
aperture_function_x = tf.expand_dims(aperture_function_coord, axis = -2)
aperture_function_y = tf.expand_dims(aperture_function_coord, axis = -1)

aperture_function_r = tf.sqrt(aperture_function_x * aperture_function_x + aperture_function_y * aperture_function_y)

aperture_function_outer_edge = ((aperture_diameter_m / 2.0) - aperture_function_r) / aperture_softness_m

aperture_function_central_obstruction = (aperture_function_r - (obstruction_diameter_m / 2.0)) / aperture_softness_m

aperture_function = tf.minimum(aperture_function_outer_edge, aperture_function_central_obstruction)
#aperture_function = aperture_function_outer_edge

aperture_function = tf.clip_by_value(aperture_function, 0.0, 1.0)

write_image(aperture_function, os.path.join(output_dir, 'aperture_function.png'))

aperture_function_complex = tf.dtypes.complex(aperture_function, tf.zeros_like(aperture_function))

aperture_function_complex *= wavefront

write_image(complex_to_image(aperture_function_complex), os.path.join(output_dir, "aperture_function_complex.png"))

psf = tf.signal.ifft2d(aperture_function_complex)

#seems I don't have this yet...
#psf = tf.signal.fftshift(psf)
# so same thing:
psf = tf.roll(psf, shift = (psf.shape[-2] // 2, psf.shape[-1] // 2), axis = (-2, -1))

# seems like i'm doing a pointless sqrt, but x * conj(x) would be doing extra stuff to compute the zero complex term - wonder if there's a better way...
psf = tf.math.square(tf.math.abs(psf))

psf /= tf.reduce_max(psf)

write_image(psf, os.path.join(output_dir, 'psf.png'))
