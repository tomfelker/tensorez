import numpy as np
import tensorflow as tf
import os
import glob
import datetime
import gc
from tensorez.util import *
from tensorez.model import *
from tensorez.bayer import *
from tensorez.fourier import *

batch_size = 5
output_path = os.path.join('data', 'synthetic')

psf_filename = os.path.join('data', 'ASICAP', 'CapObj', '2020-07-30Z', '2020-07-30-2005_0-CapObj.SER')
psf_size = 128
crop_center = (1287, 993)

true_image_filename = os.path.join('misc_images', 'STS-115_ISS_after_undocking.jpg')
#true_image_filename = os.path.join('misc_images', 'saturn20131017.jpg')
downsample = 1



def load_psfs(batch_size):
    
    os.makedirs(output_path, exist_ok = True)

    frames = []
    for frame_index in range(0, batch_size):
        frame = read_image(psf_filename, frame_index = frame_index, crop = (psf_size, psf_size), crop_center = crop_center)

        frame_max = tf.math.reduce_max(frame, axis = (-2, -3), keepdims = True)
        print(f"raw psf max is {frame_max}")

        # poke some pixels in for debugging
        if False:
            frame = frame.numpy()
            frame[0, 16, 16, 0] = 100
            frame[0, 16, 32, 1] = 100
            frame[0, 32, 32, 2] = 100
            frame[0, 48, 32, :] = 100

        frame_min = tf.math.reduce_min(frame, axis = (-2, -3), keepdims = True)
        frame = frame - frame_min
        frame_sum = tf.math.reduce_sum(frame, axis = (-2, -3), keepdims = True)
        frame = frame / frame_sum



        write_image(frame, os.path.join(output_path, f"psf{frame_index}.png"), normalize = True)


        frames.append(frame)

    frames = tf.concat(frames, axis = -4)
    print(f"frames.shape {frames.shape}")
    return frames



tf.compat.v1.enable_eager_execution()

true_psfs = load_psfs(batch_size)

true_image = read_image(true_image_filename)
true_image = tf.image.resize(true_image, (true_image.shape[-3] // downsample, true_image.shape[-2] // downsample), method = tf.image.ResizeMethod.BICUBIC)


def init_and_jam_into_model(true_image, psfs):
    model = TensoRezModel(psf_size = psfs.shape[-2], super_resolution_factor= 1, realign_center_of_mass = False, model_demosaic = False)
    model.build((batch_size, true_image.shape[-3], true_image.shape[-2], true_image.shape[-1]))
    model.estimated_image = pad_for_conv(true_image, psf_size)
    model.point_spread_functions = tf.reshape(psfs, model.point_spread_functions.shape)
    # assign instead of =, because the model build process remembered this as its trainable variable
    #todo: maybe move before build?  this whole thing is a hack
    #model.point_spread_functions.assign(tf.reshape(psfs, model.point_spread_functions.shape))
    return model

def gen_synthetic_images(true_image, psfs):

    model = init_and_jam_into_model(true_image, psfs)

    synthetic_images = model.predict_observed_images()
    return synthetic_images

synthetic_images = gen_synthetic_images(true_image, true_psfs)

for frame_index in range(0, batch_size):
    write_image(synthetic_images[frame_index,...], os.path.join(output_path, f'synthetic{frame_index}.png'))

synthetic_images_mean = tf.math.reduce_mean(synthetic_images, axis = (-4), keepdims = True)

write_image(synthetic_images[frame_index,...], os.path.join(output_path, 'prediction_mean.png'))


if True:
    print("trying to get PSF with FFT")

    observation = synthetic_images[0,...]
    #guess = true_image
    #guess = synthetic_images_mean

    reconstructed_psf = solve_for_psf(observation, guess = true_image, psf_size = psf_size)
    rms_from_true_psf = tf.reduce_mean(tf.math.square(tf.squeeze(true_psfs[0,...]) - tf.squeeze(reconstructed_psf)))
    print(f'PSF reconstructed with RMS error {rms_from_true_psf}')
    write_image(reconstructed_psf[0,...], os.path.join(output_path, "psf_guess_from_true.png"), normalize = True)

    reconstructed_psf = solve_for_psf(observation, guess = synthetic_images_mean, psf_size= psf_size)
    reconstructed_psf_unclamped = solve_for_psf(observation, guess = synthetic_images_mean, psf_size= psf_size, clamp = False)
    rms_from_true_psf = tf.reduce_mean(tf.math.square(tf.squeeze(true_psfs[0,...]) - tf.squeeze(reconstructed_psf)))
    print(f'PSF reconstructed with RMS error {rms_from_true_psf}')
    write_image(reconstructed_psf[0,...], os.path.join(output_path, "psf_guess_from_mean.png"), normalize = True)

    

if False:
    model = init_and_jam_into_model(true_image, reconstructed_psf)

    print("trying to learn one psf")

    psf_training_steps = 1000

    output_dir = os.path.join("output", "latest_psf", datetime.datetime.now().replace(microsecond = 0).isoformat().replace(':', '_'))

    psf_learning_rate = .00001

    psf_optimizer = tf.compat.v1.train.AdamOptimizer(psf_learning_rate)

    for psf_training_step in range(0, psf_training_steps):    
        with tf.GradientTape(watch_accessed_variables = False) as psf_tape:
            for var in model.psf_training_vars:
                psf_tape.watch(var)
            for var in model.losses:
                psf_tape.watch(var)
                
            model(synthetic_images)

        rms_from_true_psf = tf.reduce_mean(tf.math.square(tf.squeeze(true_psfs[0, ...]) - tf.squeeze(model.point_spread_functions)))

        print("psf step {}: loss {}, rms from true psf {}".format( psf_training_step, sum(model.losses), rms_from_true_psf))

        grads = psf_tape.gradient(model.losses, model.psf_training_vars)

        if True:
            grads = grads - 0.999 * tf.reduce_mean(grads, axis = (-4, -3))

        psf_optimizer.apply_gradients(zip(grads, model.psf_training_vars))

        model.apply_psf_physicality_constraints()

        #write_image(model.get_psf_examples(), os.path.join(output_dir, "psf_examples_latest_{}.png".format(psf_training_step)))        

        model.write_psf_debug_images(output_dir, psf_training_step)
        

