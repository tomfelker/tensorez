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
    # tf function doesn't like?
    #pygame_frame_whc = tfg.image.color_space.srgb.from_linear_rgb(pygame_frame_whc)
    pygame_frame_whc *= 255.0
    pygame_frame_whc = tf.cast(pygame_frame_whc, tf.uint8)
    pygame_frame_whc = tf.squeeze(pygame_frame_whc, axis = 0)
    pygame_frame_whc = tf.transpose(pygame_frame_whc, perm = (1, 0, 2))

    return luckiness, pygame_frame_whc


luckiness_algorithm = tensorez.luckiness.LuckinessAlgorithmLowpassAbsBandpass
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

camera.get_image_type()

pygame.init()
display = pygame.display.set_mode(size = (1024, 768), depth = 32, flags = pygame.RESIZABLE)

bit_depth = camera_info['BitDepth']

luckiness_cache = None

surface = None
running = True
while running:
    gc.collect()
    
    frame = camera.capture_video_frame()
    
    if luckiness_cache is None:
        shape = (4, frame.shape[0] // 2, frame.shape[1] // 2)

        luckiness_cache = luckiness_algorithm.create_cache(shape = shape, lights = None, average_image = None, debug_output_dir = None, debug_frames = 0, **luckiness_kwargs)

    luckiness, pygame_frame_whc = process_frame(frame, luckiness_cache, luckiness_kwargs)

    print(luckiness.numpy())

    surface = pygame.surfarray.make_surface(pygame_frame_whc)
    display.blit(surface, (0, 0))
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    


pygame.quit()
camera.close()
