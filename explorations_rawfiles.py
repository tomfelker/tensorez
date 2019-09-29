import os
import numpy as np
from PIL import Image

import rawpy

from tensorez.util import *
from tensorez.bayer import *

#lol, if you don't do this, files don't get written (since you never actually ran anything...)
tf.enable_eager_execution()


filename = os.path.join('data', 'powerline_t4i_raw', 'IMG_7054.CR2')
#filename = os.path.join('data', 'IMG_7090.CR2')
#filename = os.path.join('data', 'IMG_6607_white.CR2')

outpath = os.path.join('explorations_output', 'rawfiles')

pillow_stuff = False
if pillow_stuff:
    image = Image.open(filename)
    image_array = np.array(image)

    # for some reason, it doesn't show up...
    write_image(image_array, os.path.join(outpath, 'pillow.png'), normalize = True)

raw_stuff = False
if raw_stuff:
    raw = rawpy.imread(filename)

    raw_image_uint16 = tf.expand_dims(tf.expand_dims(raw.raw_image, axis = (-1)), axis = 0)
    raw_image_float = tf.cast(raw_image_uint16, tf.float32)

    print("raw_pattern is {}".format(raw.raw_pattern))

    write_image(raw_image_float, os.path.join(outpath, 'raw_image.png'), normalize = True)

    raw_image_bayer = apply_bayer_filter(raw_image_float, bayer_filter_tile_rggb)

    write_image(raw_image_bayer, os.path.join(outpath, 'raw_image_bayer.png'), normalize = True)

    raw_image_demosaic = apply_demosaic_filter(raw_image_bayer, demosaic_kernels_rggb)

    write_image(raw_image_demosaic, os.path.join(outpath, 'raw_image_demosaic.png'), normalize = True)

    raw_image_demosaic_curves = raw_image_demosaic / 16383.0
    black_level = tf.contrib.distributions.percentile(raw_image_demosaic_curves, 10.0)
    print("black level was {}".format(black_level))
    raw_image_demosaic_curves -= black_level
    write_image(raw_image_demosaic_curves, os.path.join(outpath, 'raw_image_demosaic_curves.png'), normalize = True)

util_test_stuff = True
if util_test_stuff:
    crop = None
    #crop = (512, 512)
    read_image_test = read_image(filename, crop = crop)
    print("done reading")
    write_image(read_image_test, os.path.join(outpath, "read_image_test.png"))

    min_color  = tf.reduce_min(read_image_test, axis = (-4, -3, -2))
    mean_color = tf.reduce_mean(read_image_test, axis = (-4, -3, -2))
    max_color  = tf.reduce_max(read_image_test, axis = (-4, -3, -2))

    print("min color is: {}".format(min_color))
    print("meancolor is: {}".format(mean_color))
    print("max color is: {}".format(max_color))
    
    
