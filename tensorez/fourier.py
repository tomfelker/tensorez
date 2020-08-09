import numpy as np
import tensorflow as tf

def pad_for_conv(t, psf_size):
    return tf.pad(t, ((0, 0), ((psf_size - 1) // 2, (psf_size + 1) // 2), ((psf_size - 1) // 2, (psf_size + 1) // 2), (0, 0)), mode = 'REFLECT')

def unpad_for_conv(t, psf_size):
    half_size = psf_size // 2
    return t[..., half_size : t.shape[-3] - half_size, half_size : t.shape[-2] - half_size, :]

def fade_edges_for_conv(t, psf_size):
    mask = tf.ones(shape = (t.shape[-3] // psf_size - 1, t.shape[-2] // psf_size - 1))
    mask = tf.pad(mask, paddings = ((1, 1), (1, 1)))
    mask = tf.expand_dims(mask, axis = -1)
    mask = tf.expand_dims(mask, axis = 0)
    mask = tf.image.resize(mask, size = (t.shape[-3], t.shape[-2]), method = tf.image.ResizeMethod.BILINEAR)
    return t * mask

def print_psf_stats(psf):
    min_val = tf.reduce_min(psf)
    max_val = tf.reduce_max(psf)
    sum_val = tf.reduce_sum(psf)
    print(f'PSF min: {min_val} max: {max_val} sum: {sum_val}')

def solve_for_psf(observation, guess, psf_size, clamp = True):
    observation = fade_edges_for_conv(observation, psf_size)
    guess = fade_edges_for_conv(guess, psf_size)

    # convert from BHWC to BCHW - not great for memory, but necessary because FFT can't stride
    observation = tf.transpose(observation, perm = (0, 3, 1, 2))
    guess = tf.transpose(guess, perm = (0, 3, 1, 2))

    # hmm, doesn't help even one ulp... maybe there isn't a double-precision fft implementation?
    if False:
        observation = tf.cast(observation, dtype = tf.dtypes.float64)
        guess = tf.cast(guess, dtype = tf.dtypes.float64)

    observation_freq = tf.signal.fft2d(tf.complex(observation, tf.cast(0, observation.dtype)))
    guess_freq = tf.signal.fft2d(tf.complex(guess, tf.cast(0, guess.dtype)))
    
    psf_freq = tf.math.divide_no_nan(observation_freq, guess_freq)

    psf = tf.signal.ifft2d(psf_freq)
    psf = tf.math.real(psf)
    psf = tf.signal.fftshift(psf, axes = (-1, -2))
    psf = tf.reverse(psf, axis = (-1, -2))
    psf = tf.transpose(psf, perm = (0, 2, 3, 1))

    if True:
        print_psf_stats(psf)

    # shrink and re-normalize
    psf = psf[..., (psf.shape[-3] - psf_size) // 2:(psf.shape[-3] + psf_size) // 2, (psf.shape[-2] - psf_size) // 2:(psf.shape[-2] + psf_size) // 2, :]
    if clamp:
        psf = tf.math.maximum(psf, 0)
        # oddly, this hurts accuracy
        if True:
            psf = psf / tf.reduce_sum(psf, axis = (-2, -3), keepdims = True)

    return psf
