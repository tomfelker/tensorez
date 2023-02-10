"""
Local Lucky Imaging

The idea is to apply lucky imaging on a per-pixel rather than per-frame basis.

The quality of the image varies not only in time, but across space, so it's
unlikely that the entire frame will be sharp at the same time - this limits
traditional lucky imaging, because that involves choosing entire frames.

Instead, we will try to determine "luckiness" for each pixel of each frame,
and then for each pixel in the result, average the corresponding luckiest
pixels in the input.

A lucky frame should agree with the average frame at low frequencies, but
contain more information than the average frame at high frequencies.  We can
use some FFT magic and hand waving to compute this.

To strictly choose from the best N frames, we would need to remember N
luckinesses and frame indices.  So we'll do that, with tensorstore and memory caching.



"""


import tensorflow as tf
import tensorstore as ts


import os
from tensorez.util import *
from tensorez.image_sequence import *
from tensorez.model import *
from tensorez.bayer import *
from tensorez.fourier import *
from tensorez.obd import *
import tensorez.drizzle
import tensorez.align
import tensorez.modified_stn
from tensorez.luckiness import *
import gc




def reduce_mean_by_top_k_scores(values, scores, k):
    # If there are ties, such that the lowest-scored value that is averaged,
    # and the lowest-scored value that isn't, have equal scores, then it's not
    # clear what we should do... averaging the values with the lower indices
    # would at least be deterministic, but still somewhat arbitrary.  Also it'd
    # be much more complex to handle all this.
    #
    # So instead: if there are such ties, than all such values will still be
    # averaged, with the divisor adjusted appropriately.

    if isinstance(scores, ts.TensorStore):
        scores_dtype = tf.as_dtype(scores.dtype.name)
        top_scores_shape = list(scores.shape)        
    else:
        scores_dtype = tf.as_dtype(scores.dtype)
        top_scores_shape = scores.shape.as_list()
    top_scores_shape[0] = 1
    top_scores_shape.append(k + 1)

    top_scores = tf.Variable(initial_value=tf.constant(value=scores_dtype.min, shape=top_scores_shape), dtype=scores_dtype)

    batch_size = values.shape[0]

    for batch_index in range(batch_size):
        _reduce_mean_by_top_k_scores_pass_1_body(top_scores, scores[batch_index : batch_index + 1, ...])

    lowest_score_to_average = top_scores[..., 0]
    del top_scores

    if isinstance(scores, ts.TensorStore):
        num_values_shape = list(scores.shape)
    else:
        num_values_shape = scores.shape.as_list()
    num_values_shape[0] = 1
    num_values = tf.Variable(initial_value=tf.zeros(shape=num_values_shape, dtype=scores_dtype))
    if isinstance(values, ts.TensorStore):
        average_dtype = tf.as_dtype(values.dtype.name)
        average_shape = list(values.shape)        
    else:
        average_shape = values.shape.as_list()
    average_shape[0] = 1
    average = tf.Variable(initial_value=tf.zeros(shape=average_shape, dtype=average_dtype))

    for batch_index in range(batch_size):
        _reduce_mean_by_top_k_scores_pass_2_body(scores[batch_index : batch_index + 1, ...], values[batch_index : batch_index + 1, ...], lowest_score_to_average, average, num_values)

    average.assign(tf.math.divide_no_nan(average, num_values))

    return average


@tf.function(jit_compile=True)
def _reduce_mean_by_top_k_scores_pass_1_body(top_scores, score):
    # sigh...
    top_scores[..., 0].assign(score)
    #top_scores.assign(tf.concat([score, top_scores[..., 0]], axis=0))

    top_scores.assign(tf.sort(top_scores, axis=-1))

@tf.function(jit_compile=True)
def _reduce_mean_by_top_k_scores_pass_2_body(score, value, lowest_score_to_average, average, num_values):
    include_in_average = tf.greater_equal(score, lowest_score_to_average)
    average.assign_add(tf.where(
        condition=include_in_average,
        x=value,
        y=0.0
    ))
    num_values.assign_add(tf.where(
        condition=include_in_average,
        x=1.0,
        y=0.0
    ))


def local_lucky_precise(
    lights,
    algorithm,
    algorithm_kwargs,
    top_fraction=.01,
    top_k=None,
    average_image = None,
    debug_output_dir = None,
    cache_dir = os.path.join('cache', 'local_lucky_precise'),
    debug_frames = 10,
    # todo:
    #drizzle = True,
    #drizzle_kwargs = {},
    #bayer = True,
    #low_memory = False
):
    if top_k is None:
        top_k = round(len(lights) * top_fraction)
        if top_k == 0:
            top_k = 1
    else:
        assert top_fraction is None
        top_fraction = top_k / len(lights)

    tensorez_steps = lights.tensorez_steps
    tensorez_steps += 'Running local_lucky_precise() with:\n'
    tensorez_steps += f'\talgorithm={algorithm}\n'
    tensorez_steps += f'\talgorithm_kwargs={algorithm_kwargs}\n'
    tensorez_steps += f'\top_k={top_k}\n'
    # todo:
    #tensorez_steps += f'\tdrizzle={drizzle}\n'
    #if drizzle:
    #    tensorez_steps += f'\tdrizzle_kwargs={drizzle_kwargs}\n'
    #tensorez_steps += f'\tbayer={bayer}\n'
    print(tensorez_steps)

    shape = hwc_to_chw(lights[0]).shape

    algorithm_cache = algorithm.create_cache(shape, lights, average_image, debug_output_dir, debug_frames, **algorithm_kwargs)

    image_store_bhwc = None
    luckiness_store_bhwc = None
    
    for image_index in range(len(lights)):
        image, dark_variance = lights.read_cooked_image(image_index, want_dark_variance=True)
  
        # todo: fancy upscaling?
      
        if image_store_bhwc is None:
            image_store_bhwc = ts.open({
                'driver': 'n5',
                'kvstore': {
                    'driver': 'file',
                    'path': os.path.join(cache_dir, 'image_store_bhwc'),
                },
                'metadata': {
                    'dataType': 'float32',
                    'dimensions': (len(lights), image.shape[-3], image.shape[-2], image.shape[-1]),
                },
                'create': True,
                'delete_existing': True,
            }).result()

        image_store_bhwc[image_index : image_index + 1, :, :, :] = image
        
        image = bhwc_to_bchw(image)
        if dark_variance is not None:
            dark_variance = bhwc_to_bchw(dark_variance)

        want_debug_images = (debug_output_dir is not None and image_index < debug_frames)

        luckiness_bchw, debug_images = algorithm.compute_luckiness(image, dark_variance, algorithm_cache, want_debug_images, **algorithm_kwargs)

        if want_debug_images:
            write_image(bchw_to_bhwc(luckiness_bchw), os.path.join(debug_output_dir, "luckiness_{:08d}.png".format(image_index)), normalize = True)
            for debug_image_name, debug_image in debug_images.items():
                write_image(bchw_to_bhwc(debug_image), os.path.join(debug_output_dir, f"{debug_image_name}_{image_index:08d}.png"))

        if luckiness_store_bhwc is None:
            luckiness_store_bhwc = ts.open({
                'driver': 'n5',
                'kvstore': {
                    'driver': 'file',
                    'path': os.path.join(cache_dir, 'luckiness_store_bhwc'),
                },
                'metadata': {
                    'dataType': 'float32',
                    'dimensions': (len(lights), luckiness_bchw.shape[-2], luckiness_bchw.shape[-1], luckiness_bchw.shape[-3]),
                },
                'create': True,
                'delete_existing': True,
            }).result()



        luckiness_store_bhwc[image_index : image_index + 1, :, :, :] = bchw_to_bhwc(luckiness_bchw)

    gc.collect()

    lucky_image_bhwc = reduce_mean_by_top_k_scores(image_store_bhwc, luckiness_store_bhwc, top_k)

    if debug_output_dir is not None:
        write_image(lucky_image_bhwc, os.path.join(debug_output_dir, 'aa_lucky_image.png'))

    return lucky_image_bhwc


