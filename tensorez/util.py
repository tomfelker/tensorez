import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg
import fnmatch
import os
import glob
import math
import datetime
import sys

#from tensorez.ser_format import *
import tensorez.ser_format as ser_format
from tensorez.bayer import *

from PIL import Image
try:
    import rawpy
except:
    print("Failed to import RawPY, needed for raw (CR2, etc.) support...")


def create_timestamped_output_dir(name):
    output_dir = os.path.join("output", name, datetime.datetime.now().replace(microsecond = 0).isoformat().replace(':', '_'))

    os.makedirs(output_dir, exist_ok = True)
    try:    
        import tee    
        tee.StdoutTee(os.path.join(output_dir, 'log.txt'), buff = 1).__enter__()
        sys.stderr = sys.stdout
    except ModuleNotFoundError:
        print("Warning: to generate log.txt, need to install tee.")
        pass

    return output_dir



def map_along_tensor_axis(func, tensor, axis, keepdims = False):
    if keepdims:
        return tf.concat(list(map((lambda unstacked: func(tf.expand_dims(unstacked, axis = axis))), tf.unstack(tensor, axis = axis))), axis = axis)
    else:
        return tf.stack(list(map(func, tf.unstack(tensor, axis = axis))), axis = axis)


# I have no idea why this doesn't seem to come with TensorFlow...
def gaussian(x, mean = 0.0, variance = 1.0):
    sigma = tf.sqrt(variance)
    return (1.0 / (sigma * tf.sqrt(2 * tf.constant(math.pi, dtype = tf.float32)))) * tf.exp(-0.5 * tf.square((x - mean) / sigma))

def radially_symmetric_psf(size, function_of_radius):
    coords = tf.linspace(-0.5, 0.5, size)
    
    coords_x = tf.expand_dims(coords, axis = -2)
    coords_y = tf.expand_dims(coords, axis = -1)
    coords_r = tf.sqrt(tf.square(coords_x) + tf.square(coords_y))
    
    psf = function_of_radius(coords_r)

    psf = psf / tf.reduce_sum(psf)
    psf = tf.expand_dims(psf, axis = -1)
    return psf

# a square, centered PSF, with standard deviation in terms of the dimension of the square
def gaussian_psf(size, standard_deviation = .1):
    variance = tf.square(standard_deviation)
    return radially_symmetric_psf(size, lambda radius : gaussian(radius, variance = variance))

# a triangle or tent kernel, with radius in terms of the dimension of the square
def tent_psf(size, radius = .5):
    return radially_symmetric_psf(size, lambda radius : tf.math.maximum(0, .5 - radius))


def gaussian_kernel_2d(size, standard_deviation):
    coords_y = tf.expand_dims(tf.linspace(-1.0 * (size[-2] // 2), 1.0 * (size[-2] - size[-2] // 2), size[-2]), axis = -1)
    coords_x = tf.expand_dims(tf.linspace(-1.0 * (size[-1] // 2), 1.0 * (size[-1] - size[-1] // 2), size[-1]), axis = -2)
    coords_r = tf.sqrt(tf.square(coords_x) + tf.square(coords_y))
    kernel = gaussian(coords_r, variance = tf.square(standard_deviation))
    kernel = kernel / tf.reduce_sum(kernel)
    return kernel

def promote_to_three_channels(image):
    if image.shape.ndims == 2:
        image = tf.expand_dims(image, axis = -1)
    if image.shape.ndims == 4:
        image = tf.squeeze(image, axis = -4)
    if image.shape[-1] == 1:
        image = tf.concat([image, image, image], axis = -1)
    return image

def crop_image(image, crop, crop_align, crop_offsets = None, crop_center = None):
    if crop is not None:
        width, height = crop
        if crop_center is not None:
            x = crop_center[0] - width // 2
            y = crop_center[1] - height // 2
        else:
            if crop_offsets is None:
                crop_offsets = (0, 0)
            x = int((image.shape[-2] - width)) // 2 + crop_offsets[0]
            y = int((image.shape[-3] - height)) // 2 + crop_offsets[1]
        
        # round so as not to mess up the bayer pattern
        x = (x // crop_align) * crop_align
        y = (y // crop_align) * crop_align
        return image[..., y : y + height, x : x + width, :]
    return image

def read_image(filename, to_float = True, srgb_to_linear = True, crop = None, crop_align = 2, crop_offsets = None, crop_center = None, color_balance = True, demosaic = True, frame_index = None):
    #print("Reading", filename, "frame " + str(frame_index) if frame_index is not None else "")
    if fnmatch.fnmatch(filename, '*.tif') or fnmatch.fnmatch(filename, '*.tiff'):
        image = Image.open(filename)        
        image = np.array(image)
        if to_float:
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.expand_dims(image, axis = -4)
            if srgb_to_linear:
                image = tfg.image.color_space.linear_rgb.from_srgb(image)

    elif fnmatch.fnmatch(filename, '*.cr2'):
        # For raw files, we will read them into a full color image, but with the bayer pattern... this way,
        # we can easily handle whatever bayer pattern

        with rawpy.imread(filename) as raw:
            # Note: there seems to be a bug where if you leave the image in numpy-land and let raw go out of scope, it silently crashes.
            # Copying to tensorflow seems to work around it.  Making a numpy copy to be doubly sure.
            image = np.copy(raw.raw_image_visible)
            image = tf.expand_dims(image, axis = -1)
            image = tf.expand_dims(image, axis = 0)
            # todo: color balance?
            # todo: black level?
            # todo: curve?
            # todo: warn about weird raw settings
            # todo, why exactly 14 bits?  just my camera?
            if to_float:
                image = tf.cast(image, tf.float32)
                
                if ((color_balance or demosaic) and (
                    chr(raw.color_desc[raw.raw_pattern[0,0]]) != 'R' or
                    chr(raw.color_desc[raw.raw_pattern[0,1]]) != 'G' or
                    chr(raw.color_desc[raw.raw_pattern[1,0]]) != 'G' or
                    chr(raw.color_desc[raw.raw_pattern[1,1]]) != 'B')):
                        print("Warning: raw file has weird bayer pattern {}".format(raw.color_desc))
                
                if color_balance:

                    scale_tile = tf.convert_to_tensor([
                        [raw.daylight_whitebalance[0], raw.daylight_whitebalance[1]],
                        [raw.daylight_whitebalance[1], raw.daylight_whitebalance[2]]
                        ], dtype = tf.float32)
                    black_tile = tf.convert_to_tensor([
                        [raw.black_level_per_channel[raw.raw_pattern[0,0]], raw.black_level_per_channel[raw.raw_pattern[0,1]]],
                        [raw.black_level_per_channel[raw.raw_pattern[1,0]], raw.black_level_per_channel[raw.raw_pattern[1,1]]]                        
                        ], dtype = tf.float32)

                    scale_tile /= 16383.0

                    scale = tf.reshape(tf.tile(scale_tile, multiples = (image.shape[-3] // 2, image.shape[-2] // 2)), image.shape)
                    black = tf.reshape(tf.tile(black_tile, multiples = (image.shape[-3] // 2, image.shape[-2] // 2)), image.shape)

                    image = (image - black) * scale

                if demosaic:
                    image = apply_demosaic_filter(image, demosaic_kernels_rggb)

    elif fnmatch.fnmatch(filename, '*.ser'):
        image, header = ser_format.read_frame(filename, frame_index = frame_index, to_float = to_float)
        if demosaic:
            if header.color_id == ser_format.ColorId.MONO:
                #print("debayering when we shouldn't, lest code below crashes?")
                #image = apply_demosaic_filter(image, demosaic_kernels_rggb)
                pass
            elif header.color_id == ser_format.ColorId.BAYER_RGGB:
                image = apply_demosaic_filter(image, demosaic_kernels_rggb)
            elif header.color_id == ser_format.ColorId.BAYER_GRBG:
                image = apply_demosaic_filter(image, demosaic_kernels_grbg)
            else:
                raise RuntimeError(f"SER file {filename} has unrecognized bayer pattern {header.color_id}")

        if to_float:
            image = tf.convert_to_tensor(image, dtype = tf.float32)            
    else:
        image = tf.io.read_file(filename)
        image = tf.io.decode_image(image)
        image = tf.expand_dims(image, axis = -4)
        if to_float:
            image = tf.cast(image, tf.float32) / 255.0
            if srgb_to_linear:
                image = tfg.image.color_space.linear_rgb.from_srgb(image)

    image = crop_image(image, crop = crop, crop_align = crop_align, crop_offsets = crop_offsets, crop_center = crop_center)
        
    return image

# todo: proper glob support, so callers don't have to do that themselves, and we can support skip and step on things like image*.cr2
class ImageSequenceReader:
    def __init__(self, filename, skip = 0, step = 1, **kwargs):
        self.filename = filename
        self.kwargs = kwargs
        self.skip = 0
        self.step = step

    def __iter__(self):
        self.current_frame_index = self.skip
        self.num_frames = 1
        if fnmatch.fnmatch(self.filename, '*.ser'):
            ser_header = ser_format.read_ser_header(self.filename)
            self.num_frames = ser_header.frame_count
        return self

    def __next__(self):
        if self.current_frame_index >= self.num_frames:
            raise StopIteration
        image = read_image(self.filename, frame_index = self.current_frame_index, **self.kwargs)
        self.current_frame_index += self.step
        return image, self.current_frame_index - 1



def write_image(image, filename, normalize = False, saturate = True, frequency_domain = False):
    if frequency_domain:
        # we assume it's indexed in channels,height,width, and is non-fftshifted, and is complex numbers.
        # we will output a human-visible image for the magnitude and the phase.
        image = chw_to_hwc(tf.signal.fftshift(image, axes = [-1, -2]))

        basename, extension = os.path.splitext(filename)

        image_log_mag = tf.math.log(tf.math.abs(image))
        write_image(image_log_mag, basename + '_log_mag' + extension, normalize = True, saturate = False, frequency_domain = False)

        image_phase = (tf.math.angle(image) + math.pi) * (math.pi / 2)
        write_image(image_phase, basename + '_phase' + extension, normalize = False, saturate = False, frequency_domain = False)
        return

    print("Writing", filename)
    if normalize:
        max_val = tf.reduce_max(image)
        print(f'max was {max_val.numpy()}, normalizing') 
        image = image / max_val
    if saturate:
        image = tf.maximum(0.0, tf.minimum(1.0, image))
    image = promote_to_three_channels(image)
    image_srgb = tfg.image.color_space.srgb.from_linear_rgb(image) * 255.0  #hmm, correct rounding?
    image_srgb_int = tf.cast(image_srgb, tf.uint8)
    image_bytes = tf.image.encode_png(image_srgb_int)
    tf.io.write_file(filename, image_bytes)

def write_sequential_image(image, path, name, sequence_num, extension = 'png', **write_image_args):
    name_with_sequence = "{}_{:08d}.{}".format(name, sequence_num, extension)
    name_with_latest = "{}_latest.{}".format(name, extension)

    if sequence_num == 0:
        history_path = os.path.join(path, 'initial', name_with_sequence)
    else:
        history_path = os.path.join(path, 'history', name, name_with_sequence)
        
    temp_path = os.path.join(path, name_with_sequence)
    latest_path = os.path.join(path, name_with_latest)
    
    write_image(image, history_path, **write_image_args)
    try:
        # hard links on windows!  Let's hope it doesn't all come crashing down... but if we do it right, it's atomic and we only wrote the file once.
        # would be best on linux, where we could link and replace in one atomic step without the temp name, but python doesn't expose this, and windows probably can't do it
        os.link(history_path, temp_path)
        os.replace(temp_path, latest_path)
        
        # and let's copy it up one also...
        temp_path = os.path.join(path, '..', name_with_sequence)
        latest_path = os.path.join(path, '..', name_with_latest)
        os.link(history_path, temp_path)
        os.replace(temp_path, latest_path)
    except Exception as e:
        # sometimes this gets access denied, because windows is lame and its filesystem won't let you be atomic
        print("Problem when moving files: {}".format(e))
    

@tf.function
def center_of_mass(image, collapse_channels = True, only_above_average = True):
    # todo: fix this - it's too complicated, and doesn't work batchwise...
    if image.shape.ndims == 4:
        image = tf.squeeze(image, axis = -4)
    
    #print("image.shape:", image.shape)

    spatial_dims = [-3, -2]

    #print("spatial_dims:", spatial_dims)

    if collapse_channels:
        image = tf.reduce_sum(image, axis = -1, keepdims = True)

    if only_above_average:
        average = tf.reduce_mean(image, keepdims = False)
        image = tf.maximum(image - average, 0)
    
    total_mass = tf.reduce_sum(image, axis = spatial_dims, keepdims = True)        
    #print("total_mass:", total_mass)
    
    ret = None
    for dim in spatial_dims:
        #print("Evaluating CoM in dim", dim)
        dim_size = image.shape[dim]
        #print("which is of size", dim_size)
        multiplier = tf.linspace(-dim_size / 2.0, dim_size / 2.0, dim_size)

        multiplier_shape = [1, 1, 1]        
        multiplier_shape[dim] = dim_size        
        #print("Multiplier shape:", multiplier_shape)
        
        multiplier_shaped = tf.reshape(multiplier, multiplier_shape)
        moments = tf.multiply(image, multiplier_shaped)
        
        com_in_dim = tf.reduce_sum(moments, axis = spatial_dims, keepdims = True) / total_mass
        #print('com_in_dim.shape', com_in_dim.shape)
        com_in_dim = tf.squeeze(com_in_dim, axis = spatial_dims)
        com_in_dim = tf.stack([com_in_dim], axis = -2)        
        #print('com_in_dim.shape', com_in_dim.shape)

        if ret is None:
            ret = com_in_dim
        else:
            ret = tf.concat([ret, com_in_dim], axis = -2)
    #print(ret)
    return ret

@tf.function
def align_by_center_of_mass(image_hwc, max_align_steps = 10, only_even_shifts = False):
    for align_step in tf.range(max_align_steps):
        image_hwc, shift, shift_axis = center_image(image_hwc, only_even_shifts = only_even_shifts)
        #tf.print("Shifted by ", shift)
        if tf.reduce_max(tf.abs(shift)) < (2 if only_even_shifts else 1):
            #tf.print("Centered after ", align_step, " steps.")
            break
        if align_step + 1 >= max_align_steps:
            tf.print("Alignment didn't converge after ", align_step + 1, " steps.")
            break
    return image_hwc
    

def pad_image(image, pad):
    if pad is not 0:
        paddings = tf.constant([[pad, pad], [pad, pad], [0, 0]])
        image = tf.pad(image, paddings)
    return image

@tf.function
def center_image(image, pad = 0, only_even_shifts = False):
    image = pad_image(image, pad)
    
    spatial_dims = [-3, -2]

    com = center_of_mass(image)

    shift = tf.squeeze(com, axis = -1)    
    shift = tf.cast(shift, tf.int32)

    #theory - only shifting by multiples of 2 may help avoid artifacts do to sensor debayering
    if only_even_shifts:
        shift = tf.bitwise.bitwise_and(shift, -2)
    
    shift = shift * -1
    #print("shift:", shift)
    shift_axis = spatial_dims
    image = tf.roll(image, shift = shift, axis = shift_axis)

    return image, shift, shift_axis


def center_image_per_channel(image, pad = 0, **kwargs):
    return map_along_tensor_axis((lambda image: center_image(image, **kwargs)[0]), image, axis = -1, keepdims = True)
    

def vector_to_graph(v, ysize = None, line_thickness = 1):
    if ysize is None:
        ysize = v.shape[0]
    
    lin = tf.linspace(1.0, 0.0, ysize);
    lin = tf.expand_dims(lin, axis = 1)
    v = tf.expand_dims(v, axis = 0)
    distances_to_line = tf.math.abs(v - lin) * tf.cast(ysize, tf.float32);
    return tf.maximum(0, tf.minimum(1, (line_thickness / 2) - distances_to_line + 1))

def adc_function_to_graph(adc, **kwargs):
    channels = tf.unstack(adc, axis = -1)
    channels = list(map((lambda channel: vector_to_graph(channel, **kwargs)), channels))
    return tf.stack(channels, axis = -1)

def center_images(images, **kwargs):
    print("centering images, shape {}".format(images.shape))
    return map_along_tensor_axis((lambda image: center_image(image, **kwargs)[0]), images, 0)

def hwc_to_chw(t):
    # hmm, doesn't work in @tf.function context... just use the more hardcoded versions below
    rank = tf.rank(t)
    perm = list(range(0, rank))
    perm[-3] = rank - 1
    perm[-2] = rank - 3
    perm[-1] = rank - 2
    return tf.transpose(t, perm = perm)

def chw_to_hwc(t):
    rank = tf.rank(t)
    perm = list(range(0, rank))
    perm[-3] = rank - 2
    perm[-2] = rank - 1
    perm[-1] = rank - 3
    return tf.transpose(t, perm = perm)

def bhwc_to_bchw(t):
    return tf.transpose(t, perm = [0, 3, 1, 2])

def bchw_to_bhwc(t):
    return tf.transpose(t, perm = [0, 2, 3, 1])

def real_to_complex(t):
    return tf.complex(t, tf.cast(0, t.dtype))

def signal_fftshift_2d(x):
    return tf.roll(x, shift = [int(x.shape[-2] // 2), int(x.shape[-1] // 2)], axis = [-2, -1])

def load_average_image(file_glob, frame_limit = None, step = 1):
    average_image = None
    image_count = 0
    for filename in glob.glob(file_glob):
        for image_hwc, frame_index in ImageSequenceReader(filename, step = step, to_float = True, demosaic = True):           

            if average_image is None:
                average_image = tf.Variable(tf.zeros_like(image_hwc))
            
            average_image.assign(average_image + image_hwc)

            image_count += 1
            if frame_limit is not None and image_count >= frame_limit:
                break

    if image_count == 0:
        raise RuntimeError(f"Couldn't load any images from '{file_glob}'.")        

    average_image.assign(average_image * (1.0 / image_count))
    return average_image, image_count

@tf.function
def image_with_zero_mean_and_unit_variance(image_bhwc):
    mean, variance = tf.nn.moments(image_bhwc, axes = (-3, -2))
    stdev = tf.sqrt(variance)
    return (image_bhwc - mean) / stdev

# similar to itertools.islice, but also supports __getitem__() and __len__()
class LimitedIterable:
    def __init__(self, iterable_with_len, limit):
        self.iterable = iterable_with_len
        self.limit = limit
    
    def __getitem__(self, index):
        if index >= self.limit:
            raise IndexError
        return self.iterable.__getitem__(index)

    def __len__(self):
        return min(self.limit, len(self.iterable))

    def __iter__(self):
        for i in range(0, len(self)):
            yield self[i]
