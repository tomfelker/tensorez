import math
import zwoasi
import simpleaudio


#import glumpy
#from OpenGL import GLUT as glut

#import matplotlib.pyplot as plt
#from matplotlib.pyplot import subplots,close
#from matplotlib import cm

import pygame

import numpy as np

import tensorez.luckiness
import tensorez.util
import tensorflow as tf
import tensorflow_graphics as tfg
import os
import gc


@tf.function
def process_frame(raw_frame_hw, luckiness_cache, luckiness_kwargs, downsample = 1, debayer = True):

    frame = tf.cast(raw_frame_hw, dtype = tf.float32)
    frame /= (1 << 16) - 1
       
    # frame is now shape bhwc, 0 to 1

    if debayer:
        even_rows = frame[0::2, :]
        odd_rows = frame[1::2, :]

        upper_left = even_rows[:, 0::2]
        upper_right = even_rows[:, 1::2]
        lower_left = odd_rows[:, 0::2]
        lower_right = odd_rows[:, 1::2]

        frame = tf.stack([upper_left, upper_right, lower_left, lower_right], axis = -1)
    else:
        # still make a channel dimension, just keep it size 1
        frame = tf.expand_dims(frame, axis = -1)

    # make batch dim
    frame = tf.expand_dims(frame, axis = 0)
    

    luckiness = luckiness_algorithm.compute_luckiness(tensorez.util.bhwc_to_bchw(frame), dark_variance_chw = None, cache = luckiness_cache, want_debug_images = False, **luckiness_kwargs)
    luckiness = tf.reduce_mean(luckiness)

    if debayer:
        # todo: other bayer paterns
        red = frame[:, :, :, 0]
        green = (frame[:, :, :, 1] + frame[:, :, :, 2]) / 2
        blue = frame[:, :, :, 3]
        pygame_frame_whc = tf.stack([red, green, blue], axis = -1)
    else:
        white = frame[:, :, :, 0]
        pygame_frame_whc = tf.stack([white, white, white], axis = -1)
    # tf function doesn't like.  need to upgrade tensorflow to avoid this, but that means i need to install linux inside windows and jesus christ
    #pygame_frame_whc = tfg.image.color_space.srgb.from_linear_rgb(pygame_frame_whc)
    pygame_frame_whc *= 255.0
    pygame_frame_whc = tf.cast(pygame_frame_whc, tf.uint8)
    pygame_frame_whc = tf.squeeze(pygame_frame_whc, axis = 0)
    pygame_frame_whc = tf.transpose(pygame_frame_whc, perm = (1, 0, 2))

    audio = generate_audio(luckiness)

    return luckiness, pygame_frame_whc, audio

@tf.function
def generate_audio(luckiness):
    # tunables    
    audio_sample_rate = 48000.0
    audio_length_seconds = .1    
    base_freq_hz = 420.0
    envelope_octaves = 3
    luckiness_per_octave = 10.0
    octave_pattern = tf.constant([0, 3], dtype = tf.float32) / 12.0
    #octave_pattern = tf.constant([0, 1, 2, 3, 4], dtype = tf.float32) / 5.0

    # shapes:
    # n means note (which of the various notes we're playing)
    # s means sample (which audio sample we're computing)
    # c means channels

    # derived constants
    audio_samples = int(audio_sample_rate * audio_length_seconds)
    freqs_octaves_n_array = []
    for octave in range(0, envelope_octaves + 1):
        freqs_octaves_n_array.append(octave_pattern + tf.cast(octave, dtype=tf.float32))
    freqs_octaves_n = tf.concat(freqs_octaves_n_array, axis = 0)

    # dynamic stuff
    freqs_octaves_n += luckiness / luckiness_per_octave
    freqs_octaves_n = tf.math.mod(freqs_octaves_n, envelope_octaves)
    
    freqs_hz_n = base_freq_hz * tf.pow(2.0, freqs_octaves_n)
    amplitudes_n = tf.maximum(0.0, 1 - tf.abs((freqs_octaves_n / envelope_octaves) * 2 - 1))

    times_seconds_s = tf.linspace(0.0, audio_samples - 1.0, audio_samples) / audio_sample_rate

    freqs_hz_ns = tf.expand_dims(freqs_hz_n, axis = 1)
    amplitudes_ns = tf.expand_dims(amplitudes_n, axis = 1)
    times_seconds_ns = tf.expand_dims(times_seconds_s, axis = 0)

    # nice sine waves - but, since sdlmixer sucks at crossfading, maybe want sawtooth wave or something so the cuts are less poppy
    #audio_ns = amplitudes_ns * tf.math.sin(times_seconds_ns * freqs_hz_ns * tf.constant(math.pi * 2, dtype=tf.float32))
    audio_ns = amplitudes_ns * tf.math.mod(times_seconds_ns * freqs_hz_ns, 1.0)

    audio_s = tf.reduce_mean(audio_ns, axis = 0)

    # do some magic so our chunk loops perfectly
    tile_fade_s = tf.abs(tf.linspace(-1.0, 1.0, audio_samples))
    audio_s = audio_s * (1.0 - tile_fade_s) + tf.roll(audio_s, shift = audio_samples // 2, axis = 0) * tile_fade_s

    audio_int16_s = tf.cast(audio_s * 32767.0, dtype = tf.int16)
    # it seems sdlmixer can't broadcast, need to stack for stereo
    #audio_int16_sc = tf.expand_dims(audio_int16_s, axis = 1)
    audio_int16_sc = tf.stack([audio_int16_s, audio_int16_s], axis = 1)

    return audio_int16_sc

#tf.config.run_functions_eagerly(True)

generate_audio(.15)

#luckiness_algorithm = tensorez.luckiness.LuckinessAlgorithmLowpassAbsBandpass
#luckiness_kwargs = dict(crossover_wavelength_pixels = 20, noise_wavelength_pixels = 2)

luckiness_algorithm = tensorez.luckiness.LuckinessAlgorithmFourierFocus
luckiness_kwargs = dict(crossover_wavelength_pixels = 20, noise_wavelength_pixels = 2)

zwoasi_library_file = os.getenv('ZWO_ASI_LIB') or 'c:/Program Files/ASIStudio/ASICamera2.dll'

try:
    zwoasi.init(zwoasi_library_file)
except:
    print('Set environment var ZWO_ASI_LIB to the path to ASICamera2.dll')
    exit()

cameras = zwoasi.list_cameras()

if len(cameras) == 0:
    print('No cameras found')
    exit()

camera = zwoasi.Camera(0)

camera_info = camera.get_camera_property()

print(camera_info)

camera.set_roi(image_type = zwoasi.ASI_IMG_RAW16)

camera.start_video_capture()

# seems to be usec
camera.set_control_value(zwoasi.ASI_EXPOSURE, 100000)

pygame.init()
display = pygame.display.set_mode(size = (1024, 768), depth = 32, flags = pygame.RESIZABLE)

pygame.mixer.init(frequency = 48000, size = -16, channels = 2)

bit_depth = camera_info['BitDepth']

luckiness_cache = None

surface = None
running = True
sound = None
while running:
    
    try:
        frame = camera.capture_video_frame()
    except zwoasi.ZWO_IOError:
        print('timeout')
        continue
    
    if luckiness_cache is None:
        shape = (4, frame.shape[0] // 2, frame.shape[1] // 2)

        luckiness_cache = luckiness_algorithm.create_cache(shape = shape, lights = None, average_image = None, debug_output_dir = None, debug_frames = 0, **luckiness_kwargs)

    luckiness, pygame_frame_whc, audio = process_frame(frame, luckiness_cache, luckiness_kwargs)

    print(luckiness.numpy())

    # tried crossfading, but SDL_mixer doesn't change volume smoothly, so it still has pops :-(
    new_sound = pygame.mixer.Sound(audio.numpy())
    new_sound.play(loops = -1)
    if sound is not None:
        sound.stop()
    sound = new_sound

    surface = pygame.surfarray.make_surface(pygame_frame_whc)
    display.blit(surface, (0, 0))
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    gc.collect()

pygame.quit()
camera.close()
