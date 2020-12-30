import tensorflow as tf
import tensorez.modified_stn as stn
import os
import os.path



def align_image(unaligned_image_bhwc, target_image_bhwc, **kwargs):
    alignment_transform = compute_alignment_transform(unaligned_image_bhwc, target_image_bhwc, **kwargs)
    aligned_image = transform_image(unaligned_image_bhwc, alignment_transform)
    return aligned_image

@tf.function
def compute_alignment_transform(
    unaligned_image_bhwc,
    target_image_bhwc,
    mask_image_bhwc,
    allow_translation = True,
    allow_rotation = False,
    allow_skew = False,
    downsample_count = 4,
    min_size = 3,
    only_even_shifts = False
    ):
    if downsample_count is 0:
        pass
    else:
        pass

def generate_mask_image(shape_bhwc, **kwargs):
    mask_image_w = generate_mask_image_1d(shape_bhwc[-2], **kwargs)
    mask_image_h = generate_mask_image_1d(shape_bhwc[-3], **kwargs)

    mask_image_b_wc = tf.reshape(mask_image_w, (1, 1, shape_bhwc[-2], 1))
    mask_image_bh_c = tf.reshape(mask_image_h, (1, shape_bhwc[-3], 1, 1))

    mask_image_bhwc = tf.multiply(mask_image_b_wc, mask_image_bh_c)

    return mask_image_bhwc


def generate_mask_image_1d(size, border_fraction = .1, dtype = tf.float32):

    mask_image_1d = tf.linspace(0.0, 1.0, size)
    mask_image_1d = tf.minimum(mask_image_1d, 1 - mask_image_1d)
    mask_image_1d = tf.greater(mask_image_1d, border_fraction)
    mask_image_1d = tf.cast(mask_image_1d, dtype)

    return mask_image_1d


def transform_image(unaligned_image_bhwc, alignment_transform):

    width = unaligned_image_bhwc.shape[-2].value
    height = unaligned_image_bhwc.shape[-3].value

    fix_scale_x = width / (width - 1.0)
    fix_scale_y = height / (height - 1.0)
    fix_scale = tf.constant([
        [fix_scale_x, fix_scale_x, 1.0],
        [fix_scale_y, fix_scale_y, 1.0]
    ])

    fix_bias = tf.constant([
        [0.0, 0.0, 1.0 / (width - 1)],
        [0.0, 0.0, 1.0 / (height - 1)]
    ])

    fixed_alignment_transform = tf.add(tf.multiply(alignment_transform, fix_scale), fix_bias)

    return stn.spatial_transformer_network(unaligned_image_bhwc, alignment_transform)
    #return stn.spatial_transformer_network(unaligned_image_bhwc, fixed_alignment_transform)

def alignment_loss(unaligned_image_bhwc, target_image_bhwc, mask_image_bhwc):
    return tf.math.reduce_mean(tf.math.square(tf.math.multiply((unaligned_image_bhwc - target_image_bhwc), mask_image_bhwc)), axis = (-3, -2, -1), keepdims = True)
     

