import numpy as np
import tensorflow as tf
import os
import glob
from tensorstax_util import *
from tensorstax_model import *
from tensorstax_bayer import *


tf.enable_eager_execution()


def generate_synthetic_data(image_filename, output_path, **kwargs):

    true_image = image_load(image_filename)

    model = TensorstaxModel(kwargs)
