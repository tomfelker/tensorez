import numpy as np
import tensorflow as tf
import math

from tensorez.util import *
from tensorez.bayer import *
from tensorez.fourier import *

# This is heavily inspired by papers like:
#
# http://is.tuebingen.mpg.de/fileadmin/user_upload/files/publications/ICCP2009-Harmeling_5649%5B0%5D.pdf
# https://www.aanda.org/articles/aa/full_html/2011/07/aa13955-09/aa13955-09.html
#
# and their example code:
#
# http://webdav.is.mpg.de/pixel/obd/
# http://webdav.is.mpg.de/pixel/obd/obd.zip
#
#

# In general, their names => my names:
#   x => estimated_image
#   f => psf
#   y => observed_image

# Since tensorflow doesn't support FFTs on any but the last-most dimensions, everything in here will buck
# the normal tradition, and instead have dimensions of (channel, height, width).  Use tensorez.util.hwc_to_chw() etc.

# hmm, should this be more of a class?

# corresponds to obd()
# returns estimated_image, estimated_psf
@tf.function
def obd_step(
    estimated_image,
    observed_image,
    psf_shape,
    clip = tf.constant(255/256),
    psf_iterations = tf.constant(1000),
    estimated_image_iterations = tf.constant(1),
    tolerance = tf.constant(1e-6),
    estimated_image_update_power = tf.constant(.03)
    ):
    
    # solve (as accurately as possible) for the best all-positive-valued estimated_psf
    # to minimize the difference between
    # prev_estimated_image conv estimated_psf 
    # and
    # observed_image
    #
    # The papers say to use LBFGS-B, but the example code just uses the same obd multiplicative update
    # but iterates it 40x instead of 1x.  Presumably we can't just use inverse FFT because that would give
    # negative coefficients in cases where the prev_estimated_image is too "fuzzy" and so estimated_psf would
    # need to "sharpen" it to account for a sharper observation.

    # initial psf is just uniform
    estimated_psf = tf.constant(1 / (psf_shape[-2] * psf_shape[-1]), shape = psf_shape)

    estimated_psf = obd_update(estimated_psf, estimated_image, observed_image, iterations = psf_iterations, clip = clip)

    sum_estimated_psf = tf.reduce_sum(estimated_psf, axis = (-2, -1), keepdims = True)
    estimated_psf *= 1 / sum_estimated_psf
    #not sure why we do this:
    estimated_image *= sum_estimated_psf

    estimated_image = obd_update(estimated_image, estimated_psf, observed_image, iterations = estimated_image_iterations, clip = clip, update_power = estimated_image_update_power)

    return estimated_image, estimated_psf

def obd_update(f, x, y, iterations, clip, update_power = None, tolerance = tf.constant(1e-8)):
    m = tf.cast(y < clip, y.dtype)
    y = tf.multiply(y, m)
    for i in range(0, iterations):
        ytmp = tf.math.maximum(0.0, conv2d_fourier(x, f))
        ytmp = tf.math.multiply(ytmp, m)
        nom = tf.math.maximum(0.0, conv2d_transpose_fourier(x, y))
        denom = tf.math.maximum(0.0, conv2d_transpose_fourier(x, ytmp))        
        factor = (nom + tolerance) / (denom + tolerance)

        # just a thought, maybe we're still learning too fast
        if update_power is not None:
            factor = tf.math.pow(factor, update_power)

        f = tf.math.multiply(f, factor)
    return f

# computes x conv f
# using fourier transforms
def conv2d_fourier(x, f):
    
    # want f to be smaller, if not, swap args
    if x.shape[-1] < f.shape[-1]:
        return conv2d_fourier(f, x)

    # y = ifft2(fft2(x) .* fft2(f, sx(1), sx(2)));
    # y = cnv2slice(y, sf(1):sx(1), sf(2):sx(2));

    f_pad = pad_zeros_after_2d(f, x.shape)

    print(f'FFT shape x: {x.shape}')
    print(f'FFT shape f_pad: {f_pad.shape}')
    
    x_freq = tf.signal.fft2d(real_to_complex(x))
    f_freq = tf.signal.fft2d(real_to_complex(f_pad))
    
    y_freq = tf.math.multiply(x_freq, f_freq)

    print(f'IFFT shape y_freq: {y_freq.shape}')

    y_pad = tf.signal.ifft2d(y_freq)

    # hmm, hope i'm not off-by-one
    y = y_pad[..., f.shape[-2] - 1:x.shape[-2], f.shape[-1] - 1:x.shape[-1]]
    
    # not sure what cnv2slice is trying to do, it seems to shrink y so it's smaller than x by the size of f, but only on the top row?
    # also, not sure how/why they didn't take the real part... seems matlab/octave have some magic for detecting that case?
    y = tf.math.real(y)

    return y

# hmm, i think this is a case where 'transpose' also means 'inverse'?
def conv2d_transpose_fourier(x, y):
    # hmm, i wonder why they don't just do the same recursive call thing when switching sizes?
    if x.shape[-1] >= y.shape[-1]:
        #   f = cnv2slice(ifft2(conj(fft2(x)).*fft2(cnv2pad(y, sf))), 1:sf(1), 1:sf(2));

        print(f'FFT shape x: {x.shape}')

        x_freq = tf.signal.fft2d(real_to_complex(x))
        
        y_pad = pad_zeros_before_2d(y, x.shape)

        print(f'FFT shape y_pad: {y_pad.shape}')

        y_freq = tf.signal.fft2d(real_to_complex(y_pad))

        product = tf.math.multiply(tf.math.conj(x_freq), y_freq)

        print(f'IFFT shape product: {product.shape}')

        f_pad = tf.signal.ifft2d(product)

        #f_shape = x.shape - y.shape + 1
        f_shape = (
            x.shape[-4],
            x.shape[-3],
            x.shape[-2] - y.shape[-2] + 1,
            x.shape[-1] - y.shape[-1] + 1
        )

        f = discard_after_2d(f_pad, f_shape)
        return tf.math.real(f)
    else:
        # sf = sy + sx - 1;
        # f = ifft2(conj(fft2(x, sf(1), sf(2))).*fft2(cnv2pad(y, sx), sf(1), sf(2)));

        #f_shape = y.shape + x.shape - 1
        f_shape = (
            y.shape[-4],
            y.shape[-3],
            y.shape[-2] + x.shape[-2] - 1,
            y.shape[-1] + x.shape[-1] - 1
        )

        x_pad = pad_zeros_after_2d(x, f_shape)

        print(f'FFT shape x_pad: {x_pad.shape}')

        x_freq = tf.signal.fft2d(real_to_complex(x_pad))

        y_pad = pad_zeros_before_2d(y, f_shape)
        # hrm this makes no sense, not sure about padding

        print(f'FFT shape y_pad: {y_pad.shape}')

        y_freq = tf.signal.fft2d(real_to_complex(y_pad))

        product = tf.math.multiply(tf.math.conj(x_freq), y_freq)

        print(f'IFFT shape product: {product.shape}')

        f = tf.signal.ifft2d(product)

        return tf.math.real(f)


def centered_pad_2d(t, shape, **kwargs):
    half_extra = ((shape[-2] - t.shape[-2]) // 2, (shape[-1] - t.shape[-1]) // 2)
    return tf.pad(t, paddings = ((0, 0), (0, 0), (half_extra[-2], shape[-2] - half_extra[-2]), (half_extra[-1], shape[-1] - half_extra[-1])) **kwargs)

def pad_zeros_after_2d(t, shape, **kwargs):
    return tf.pad(t, paddings = ((0, 0), (0, 0), (0, shape[-2] - t.shape[-2]), (0, shape[-1] - t.shape[-1])), **kwargs)
    
def pad_zeros_before_2d(t, shape, **kwargs):
    return tf.pad(t, paddings = ((0, 0), (0, 0), (shape[-2] - t.shape[-2], 0), (shape[-1] - t.shape[-1], 0)), **kwargs)

def discard_after_2d(t, shape):
    return t[..., 0:shape[-2], 0:shape[-1]]

def discard_before_2d(t, shape):
    return t[..., -shape[-2]:0, -shape[-1]:0]

