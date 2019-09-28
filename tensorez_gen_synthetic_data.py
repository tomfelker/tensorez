import numpy as np
import tensorflow as tf
import os
import glob
from tensorez.util import *
from tensorez.model import *
from tensorez.bayer import *


tf.enable_eager_execution()


def generate_synthetic_data(image_filename, output_path, **kwargs):

    true_image = image_load(image_filename)

    model = TensoRezModel(kwargs)
