import numpy as np
import tensorflow as tf
import os
import glob
import datetime
import math
import stn
from tensorez.model import *
from tensorez.util import *

tf.compat.v1.enable_eager_execution()
tf.set_random_seed(0)

output_dir = os.path.join('output', 'explorations', 'psf')

#before getting too deep, let's just init these more reasonably:


write_image(gaussian_psf(64), os.path.join(output_dir, "psf_guess.png"), normalize = True)


# all the below is cool, but probably not useful

density_fudge = .0000003


# pretend telescope stats
aperture_diameter_m = 280e-3
obstruction_diameter_m = aperture_diameter_m * 95e-3 / 280e-3
focal_length_m = 2800e-3
barlow_factor = 2
sensor_pixel_size_m = 4.3e-6
wavelengths_m = tf.convert_to_tensor([700e-9, 530e-9, 470e-9])


effective_focal_length_m = focal_length_m * barlow_factor


# the size of the PSF, for FFT purposes
psf_px_full = 1024

# the size that we output / use...
psf_px_cropped = 64


sensor_pixel_angle_rad = sensor_pixel_size_m / effective_focal_length_m
print(f"Pixels are {sensor_pixel_angle_rad * 180 / math.pi * 60 * 60} SoA")

aperture_function_spatial_frequencies = psf_px_full * sensor_pixel_angle_rad / wavelengths_m

# when doing the FFT, the aperture should appear to be this many pixels large, so that the result will have the given sensor_pixel_size (i think)
aperture_diameter_px = aperture_function_spatial_frequencies * aperture_diameter_m

print(f"aperture diameter in px would be:  {aperture_diameter_px}")


# the aperture function, zoomed in such that the aperture fully fills the tensor
aperture_function_size_cropped = 1024






#Kolmogorov something something
# no_noise_below r_0 something something
def turbulence_2d(size, no_noise_at_scale = 1):
    turbulence = tf.random.normal(shape = (1, 1, 1, 1))
    steps = math.ceil(math.log(size, 2))
    for step in range(0, steps):
        # why is this hard? just want to upsample
        print("resizing to {}".format(turbulence.shape[-2]))
        turbulence = tf.image.resize(turbulence, size = (turbulence.shape[-2] * 2, turbulence.shape[-3] * 2), method = tf.image.ResizeMethod.BICUBIC)
        if steps - step - 1 >= no_noise_at_scale:
            turbulence += tf.random.normal(shape = turbulence.shape)
    return turbulence[0:size, 0:size]
        
write_image(turbulence_2d(512), os.path.join(output_dir, "turbulence_test.png"), normalize = True)

def complex_to_image(complex_tensor):
    return tf.stack([tf.math.real(complex_tensor) + .5, tf.math.imag(complex_tensor) + .5, tf.zeros(complex_tensor.shape)], axis = -1)

#write_image(complex_to_image(wavefront), os.path.join(output_dir, "wavefront.png"))

aperture_softness_px = 1
aperture_softness_m = aperture_diameter_m / aperture_function_size_cropped * aperture_softness_px

aperture_function_coord = tf.linspace(-aperture_diameter_m / 2.0, aperture_diameter_m / 2.0, num = aperture_function_size_cropped)
aperture_function_x = tf.expand_dims(aperture_function_coord, axis = -2)
aperture_function_y = tf.expand_dims(aperture_function_coord, axis = -1)

aperture_function_r = tf.sqrt(aperture_function_x * aperture_function_x + aperture_function_y * aperture_function_y)

aperture_function_outer_edge = ((aperture_diameter_m / 2.0) - aperture_function_r) / aperture_softness_m

aperture_function_central_obstruction = (aperture_function_r - (obstruction_diameter_m / 2.0)) / aperture_softness_m

aperture_function = tf.minimum(aperture_function_outer_edge, aperture_function_central_obstruction)

aperture_function = tf.clip_by_value(aperture_function, 0.0, 1.0)

aperture_function = tf.expand_dims(aperture_function, axis = -1)
aperture_function = tf.expand_dims(aperture_function, axis = -4)


write_image(aperture_function, os.path.join(output_dir, 'aperture_function.png'))

# some noise with a certain power spectrum, Kolmogorov magic
turbulence = turbulence_2d(aperture_function_size_cropped)


psfs = []
for channel in range(0, wavelengths_m.shape[0]):

    
    
    # some more magic, to turn it into the difference in optical path length
    turbulence_opd = turbulence * density_fudge

    # now as a phase delay, probably ignoring a 2pi factor that could be baked into fudge
    turbulence_phase = turbulence_opd / wavelengths_m[channel]
   

    inv_scale = psf_px_full / aperture_diameter_px[channel]
    aperture_function_scaled = stn.spatial_transformer_network(aperture_function, [[ inv_scale, 0, 0, 0, inv_scale, 0]], [psf_px_full, psf_px_full])
    turbulence_scaled = stn.spatial_transformer_network(turbulence_phase, [[inv_scale, 0, 0, 0, inv_scale, 0]], [psf_px_full, psf_px_full])

    wavefront = tf.exp(tf.dtypes.complex(tf.zeros_like(turbulence_scaled), turbulence_scaled))
    
    aperture_function_complex = tf.dtypes.complex(aperture_function_scaled, tf.zeros_like(aperture_function_scaled))
    aperture_function_complex *= wavefront

    #fft doesn't like extra dims
    aperture_function_complex = tf.squeeze(aperture_function_complex, axis = -4)
    aperture_function_complex = tf.squeeze(aperture_function_complex, axis = -1)
    
    write_image(complex_to_image(aperture_function_complex), os.path.join(output_dir, f"aperture_function_complex_ch{channel}.png"))

    psf = tf.signal.ifft2d(aperture_function_complex)

    #seems I don't have this yet...
    #psf = tf.signal.fftshift(psf)
    # so same thing:
    psf = tf.roll(psf, shift = (psf.shape[-2] // 2, psf.shape[-1] // 2), axis = (-2, -1))
    
    # seems like i'm doing a pointless sqrt, but x * conj(x) would be doing extra stuff to compute the zero complex term - wonder if there's a better way...
    psf = tf.math.square(tf.math.abs(psf))
    
    psfs.append(psf)
psfs = tf.stack(psfs, axis = -1)



psfs /= tf.reduce_max(psfs)

write_image(psfs, os.path.join(output_dir, 'psf.png'))
