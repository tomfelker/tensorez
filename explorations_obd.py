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
from tensorez.obd import *


file_glob = os.path.join('data', 'firecapture_examples', 'jup', 'Jup_fixedbayer.ser')

psf_size = 32
psf_shape = (1, 3, psf_size, psf_size)

estimated_image = None
for filename in glob.glob(file_glob):
    for image_hwc in ImageSequenceReader(filename, to_float = True, demosaic = True):
        image = hwc_to_chw(image_hwc)
        if estimated_image is None:
            estimated_image = hwc_to_chw(pad_for_conv(image_hwc, psf_size))
        else:
            estimated_image = obd_step(estimated_image, image, psf_shape)      

write_image(chw_to_hwc(estimated_image), "woot.png", normalize = True)