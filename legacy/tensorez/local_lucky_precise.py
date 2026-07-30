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
luckinesses and frame indices.  So we'll do that, with mmap and memory caching.



"""


import tensorflow as tf

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
import mmap

if getattr(mmap, 'MADV_COLD', None) is None:
    mmap.MADV_COLD = 20
if getattr(mmap, 'MADV_PAGEOUT', None) is None:
    mmap.MADV_PAGEOUT = 21


def reduce_mean_by_top_k_scores(values, scores, k):
    # If there are ties, such that the lowest-scored value that is averaged,
    # and the lowest-scored value that isn't, have equal scores, then it's not
    # clear what we should do... averaging the values with the lower indices
    # would at least be deterministic, but still somewhat arbitrary.  Also it'd
    # be much more complex to handle all this.
    #
    # So instead: if there are such ties, than all such values will still be
    # averaged, with the divisor adjusted appropriately.

    scores_dtype = tf.as_dtype(scores.dtype)
    top_scores_shape = list(scores.shape)
    top_scores_shape[0] = 1
    top_scores_shape.append(k)

    #top_scores = tf.Variable(initial_value=tf.constant(value=scores_dtype.min, shape=top_scores_shape), dtype=scores_dtype)

    # it's important that these values be unique across the last dim, or else pass 1 won't separate them
    min_score = tf.math.reduce_min(scores[0:k, ...])
    unique_score_decrements = tf.linspace(start=1.0, stop=2.0, num=k)
    unique_min_scores = min_score - tf.abs(min_score) * unique_score_decrements
    unique_min_scores = tf.broadcast_to(unique_min_scores, shape=top_scores_shape)
    top_scores = tf.Variable(initial_value=unique_min_scores)


    batch_size = values.shape[0]

    for batch_index in range(batch_size):
        print(f"reduce_mean_by_top_k_scores() pass 1 image {batch_index+1} of {batch_size}")
        _reduce_mean_by_top_k_scores_pass_1_body(top_scores, scores[batch_index : batch_index + 1, ...])

    lowest_score_to_average = tf.math.reduce_min(top_scores, axis=-1)
    del top_scores

    num_values_shape = list(scores.shape)
    num_values_shape[0] = 1
    num_values = tf.Variable(initial_value=tf.zeros(shape=num_values_shape, dtype=scores_dtype))
    
    average_dtype = tf.as_dtype(values.dtype)
    average_shape = list(values.shape)
    average_shape[0] = 1

    average = tf.Variable(initial_value=tf.zeros(shape=average_shape, dtype=average_dtype))

    for batch_index in range(batch_size):
        print(f"reduce_mean_by_top_k_scores() pass 2 image {batch_index+1} of {batch_size}")
        _reduce_mean_by_top_k_scores_pass_2_body(scores[batch_index : batch_index + 1, ...], values[batch_index : batch_index + 1, ...], lowest_score_to_average, average, num_values)

    average.assign(tf.math.divide_no_nan(average, num_values))

    return average


@tf.function(jit_compile=True)
def _reduce_mean_by_top_k_scores_pass_1_body(top_scores, score):
    
    # could do this approach, but i think i'd need some meshgrid stuff to make the indices
    #lowest_top_score_indices = tf.math.argmin(top_scores, axis=-1, output_type=tf.int32)
    #top_scores.scatter_nd_update(indices=lowest_top_score_indices, updates=score)

    # this approach is fine, though, as long as we don't mind overcounting ties (which we do handle later).
    # actually not fine... if score matches one of the non-lowest entries already in top_scores, we'll lose the uniqueness.
    # then we'll be adding too few things below...
    score = tf.expand_dims(score, axis=-1)

    lowest_top_score = tf.math.reduce_min(top_scores, axis=-1, keepdims=True)
    is_lowest_top_score = tf.math.equal(top_scores, lowest_top_score)

    score_already_in_top_scores = tf.math.equal(top_scores, score)
    score_already_in_top_scores = tf.math.reduce_any(score_already_in_top_scores, axis=-1, keepdims=True)

    insert_score_here = tf.math.logical_and(is_lowest_top_score, tf.math.logical_not(score_already_in_top_scores))

    top_scores.assign(tf.where(condition=insert_score_here, x=score, y=lowest_top_score))

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

    os.makedirs(cache_dir, exist_ok=True)
    
    for image_index in range(len(lights)):
        print(f"Computing luckiness for image {image_index+1} of {len(lights)}")

        image, dark_variance = lights.read_cooked_image(image_index, want_dark_variance=True)
  
        # todo: fancy upscaling?
      
        if image_store_bhwc is None:   
            image_store_filename = os.path.join(cache_dir, 'image_store_bhwc.npy')
            if os.path.exists(image_store_filename):
                os.unlink(image_store_filename)
            image_store_bhwc = np.lib.format.open_memmap(
                filename=image_store_filename,
                mode='w+',
                dtype=np.float32,
                shape=(len(lights), image.shape[-3], image.shape[-2], image.shape[-1]),
            )
            image_store_bhwc._mmap.madvise(mmap.MADV_SEQUENTIAL)                

        image_store_bhwc[image_index : image_index + 1, :, :, :] = image
        
        image_store_bhwc._mmap.madvise(mmap.MADV_DONTNEED)
        image_store_bhwc._mmap.madvise(mmap.MADV_COLD)
        image_store_bhwc._mmap.madvise(mmap.MADV_PAGEOUT)

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
            luckiness_store_filename = os.path.join(cache_dir, 'luckiness_store_bhwc.npy')
            if os.path.exists(luckiness_store_filename):
                os.unlink(luckiness_store_filename)
            luckiness_store_bhwc = np.lib.format.open_memmap(
                filename=luckiness_store_filename,
                mode='w+',
                dtype=np.float32,
                shape=(len(lights), luckiness_bchw.shape[-2], luckiness_bchw.shape[-1], luckiness_bchw.shape[-3]),
            )
            luckiness_store_bhwc._mmap.madvise(mmap.MADV_SEQUENTIAL)

        gc.collect()
        luckiness_store_bhwc[image_index : image_index + 1, :, :, :] = bchw_to_bhwc(luckiness_bchw)
        luckiness_store_bhwc._mmap.madvise(mmap.MADV_DONTNEED)
        luckiness_store_bhwc._mmap.madvise(mmap.MADV_COLD)
        luckiness_store_bhwc._mmap.madvise(mmap.MADV_PAGEOUT)

    gc.collect()

    print(f"Collecting top {top_k}")
    lucky_image_bhwc = reduce_mean_by_top_k_scores(image_store_bhwc, luckiness_store_bhwc, top_k)

    if debug_output_dir is not None:
        write_image(lucky_image_bhwc, os.path.join(debug_output_dir, f'aa_lucky_image_top_{top_k}_of_{len(lights)}.png'))

    return lucky_image_bhwc


