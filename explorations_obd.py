import numpy as np
import tensorflow as tf
import os
import glob
import datetime
import gc
import sys
from tensorez.util import *
from tensorez.model import *
from tensorez.bayer import *
from tensorez.fourier import *
from tensorez.obd import *


file_glob = os.path.join('data', 'firecapture_examples', 'jup', 'Jup_fixedbayer.ser')

psf_size = 128

output_dir = os.path.join("output", "latest_obd", datetime.datetime.now().replace(microsecond = 0).isoformat().replace(':', '_'))

os.makedirs(output_dir, exist_ok = True)
try:    
    import tee    
    tee.StdoutTee(os.path.join(output_dir, 'log.txt'), buff = 1).__enter__()
    sys.stderr = sys.stdout
except ModuleNotFoundError:
    print("Warning: to generate log.txt, need to install tee.")
    pass


average_image = None
image_count = 0
for filename in glob.glob(file_glob):
    for image_hwc in ImageSequenceReader(filename, to_float = True, demosaic = True):
        image_count += 1
        if average_image is None:
            average_image = image_hwc
        else:
            average_image += image_hwc
average_image *= 1 / image_count
write_image(average_image, os.path.join(output_dir, 'average_image.png'))

psf_shape = (1, 3, psf_size, psf_size)

estimated_image = None
step = 0
epoch = 0
while True:
    for filename in glob.glob(file_glob):
        for image_hwc in ImageSequenceReader(filename, to_float = True, demosaic = True):
            image = hwc_to_chw(image_hwc)
            
            if estimated_image is None:
                estimated_image = hwc_to_chw(pad_for_conv(average_image, psf_size))
            
            estimated_image, estimated_psf = obd_step(estimated_image, image, psf_shape)    

            write_every = 1 << epoch
            if step % write_every == 0:
                write_sequential_image(chw_to_hwc(estimated_image), output_dir, 'estimated_image', step, normalize = False)
                write_sequential_image(chw_to_hwc(estimated_psf), output_dir, 'estimated_psf', step, normalize = True)

            step += 1
    epoch += 1
