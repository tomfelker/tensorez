'''
drizzle.py - a python / Tensorflow implementation of the DRIZZLE algorithm
by Tom Felker
code based on and inspired by this implementation of Spatial Transformer Network (MIT license, by Kevin Zakka)
    - https://github.com/kevinzakka/spatial-transformer-network 
    - itself based on 'Spatial Transformer Networks', Jaderberg et. al,
        - https://arxiv.org/abs/1506.02025
and with reference to the DRIZZLE paper by Fruchter, A. S.; Hook, R. N.:
    - https://arxiv.org/pdf/astro-ph/9808087.pdf
    - DOI 10.1086/338393
'''

import tensorflow as tf

'''
A lot of one-letter variable names will mean something like:
    b: batch
    h: height
    w: width
    c: channel
    i: coordinate in input image
    o: coordinate in output image
'''

@tf.function
def drizzle(input_image_bhwc, theta, upscale = 2, pixfrac = .5, out_shape_hw=None, inputspace_weights_bhwc = None, outputspace_weights_bhwc = None, supersample = 4):

    b = tf.shape(input_image_bhwc)[0]
    in_h = tf.shape(input_image_bhwc)[1]
    in_w = tf.shape(input_image_bhwc)[2]
    
    # todo: add extra row for easier jacobian stuff
    theta = tf.reshape(theta, [b, 2, 3])

    if out_shape_hw is None:
        out_h = in_h * upscale
        out_w = in_w * upscale
    else:
        assert(upscale is None)
        out_h = out_shape_hw[0]
        out_w = out_shape_hw[1]

    grid_bihw = affine_grid_generator(out_h, out_w, theta)

    if supersample != 1:
        # TODO: This is doing it numerically, and computing the jacobian at every pixel.
        # that'll be helpful if we have per-pixel shifts (geometric distortion, local alignment, that sort of thing)
        # but it's actually completely pointless for affine transforms, since the valaues will actually be the same for each pixel.
        # so, in that case, we could save some memory and time by just giving a single value that can be broadcasted as needed.
        jacobian_biohw = jacobian_from_grid(grid_bihw)
    else:
        # not needed when we're not supersampling.
        jacobian_biohw = None

    if inputspace_weights_bhwc is None:
        inputspace_weights_bhwc = tf.ones_like(input_image_bhwc)
    else:
        input_image_bhwc *= inputspace_weights_bhwc
    
    output_image_bhwc = drizzle_sampler(input_image_bhwc, grid_bihw, pixfrac, supersample, jacobian_biohw)

    output_weights_bhwc = drizzle_sampler(inputspace_weights_bhwc, grid_bihw, pixfrac, supersample, jacobian_biohw)

    if outputspace_weights_bhwc is not None:
        output_weights_bhwc *= outputspace_weights_bhwc

    return output_image_bhwc, output_weights_bhwc


def get_pixel_value(image_bhwc, xi_coords_bhw, yi_coords_bhw):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - image_bhwc: tensor of shape (B, H, W, C)
    - xi_coords_bhw: for each output pixel (shape: bhw), which input x coord should we get
    - yi_coords_bhw: for each output pixel (shape: bhw), which input y coord should we get

    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(xi_coords_bhw)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, yi_coords_bhw, xi_coords_bhw], 3)

    return tf.gather_nd(image_bhwc, indices)


def affine_grid_generator(out_h, out_w, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.

    Input
    -----
    - out_h: desired height of grid/output. Used
      to downsample or upsample.

    - out_w: desired width of grid/output. Used
      to downsample or upsample.

    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.

    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.

    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid

    # x[x_o] means, for output pixels in column x_o, what x coordinate (normalized -1 to 1) from the input image should we sample?
    # we choose our sample points at the center of the output pixels.  (This doesn't match what STN did.)
    x = ((tf.range(0, out_w, dtype = tf.float32) + 0.5) * 2.0 / tf.cast(out_w, tf.float32)) - 1.0
    y = ((tf.range(0, out_h, dtype = tf.float32) + 0.5) * 2.0 / tf.cast(out_h, tf.float32)) - 1.0
    
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # this comment is a lie:
    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, out_h, out_w])

    return batch_grids

def jacobian_from_grid(grid_bihw):
    out_w = grid_bihw.shape[-1]
    out_h = grid_bihw.shape[-2]

    # we have to pad so we don't get weird edge effects
    padded_grid_bihw = tf.pad(grid_bihw, paddings = ((0, 0), (0, 0), (1, 1), (1, 1)), mode = 'REFLECT')

    # but we need to keep track of whether our input coordinates were valid, or just copies made by the padding operation - in which case,
    # we'll need to weight jacobian twice as heavily to compensate for the 0 derivative on this half.
    # valid_bihw is 1 wherever we have a valid input coordinate, and 0 where the input coordinate is just a copy (from the above padding)
    valid_bihw = tf.ones(shape = (1, 1, out_h, out_w))
    # and this is padded in the same way as the input
    padded_valid_bihw = tf.pad(valid_bihw, paddings = ((0, 0), (0, 0), (1, 1), (1, 1)), mode = 'CONSTANT', constant_values = 0)

    # these are copies of grid, shifted in each direction.  Read oxm1 as "output x minus 1"
    padded_grid_oxm1_bihw = tf.roll(padded_grid_bihw, shift = -1, axis = -1)
    padded_grid_oxp1_bihw = tf.roll(padded_grid_bihw, shift =  1, axis = -1)
    padded_grid_oym1_bihw = tf.roll(padded_grid_bihw, shift = -1, axis = -2)
    padded_grid_oyp1_bihw = tf.roll(padded_grid_bihw, shift =  1, axis = -2)

    # these are copies of valid, shifted in each direction.  Read oxm1 as "output x minus 1"
    padded_valid_oxm1_bihw = tf.roll(padded_valid_bihw, shift = -1, axis = -1)
    padded_valid_oxp1_bihw = tf.roll(padded_valid_bihw, shift =  1, axis = -1)
    padded_valid_oym1_bihw = tf.roll(padded_valid_bihw, shift = -1, axis = -2)
    padded_valid_oyp1_bihw = tf.roll(padded_valid_bihw, shift =  1, axis = -2)

    # the denominators account for how many valid coordinates we had (which should be 1 or 2)

    # read as "change in input coordinate i per change in output x coordinate"
    di_over_dxo_bihw = (padded_grid_oxp1_bihw - padded_grid_oxm1_bihw) / ((padded_valid_oxp1_bihw + padded_valid_oxm1_bihw) / out_w)
    # read as "change in input coordinate i per change in output y coordinate"
    di_over_dyo_bihw = (padded_grid_oyp1_bihw - padded_grid_oym1_bihw) / ((padded_valid_oyp1_bihw + padded_valid_oym1_bihw) / out_h)
    
    # could also have named this di_over_do_biohw, which you could read as "change in input coordinate i per change in output coordinate o"
    padded_jacobian_biohw = tf.stack([di_over_dxo_bihw, di_over_dyo_bihw], axis = 2)

    # unpad
    jacobian_biohw = padded_jacobian_biohw[:, :, :, 1 : -1, 1 : -1]

    return jacobian_biohw

def drizzle_sampler(input_bhwc, normalized_coords_bihw, pixfrac, supersample, jacobian_biohw):
    if supersample == 1:
        return drizzle_sample_centers(input_bhwc, normalized_coords_bihw, pixfrac)

    b = normalized_coords_bihw.shape[0]
    oh = normalized_coords_bihw.shape[2]
    ow = normalized_coords_bihw.shape[3]
    c = input_bhwc.shape[-1]

    # how much we perturb our output normalized coords
    ssx = ((tf.range(0, supersample, dtype = tf.float32) + 0.5) / supersample - .5) / ow
    ssy = ((tf.range(0, supersample, dtype = tf.float32) + 0.5) / supersample - .5) / oh

    # s and t are the x,y coordinates within the supersample grid

    output_perturbation_sto = tf.stack(tf.meshgrid(ssx, ssy), axis = -1)
    
    #tf.print(output_perturbation_sto)

    # what i kinda want is:
    # for s in range(0, supersample)
    #   for t in range(0, supersample)
    #       for o in (0, 1)
    #           normalized_coords_bihwst[:,:,:,:,s,t] = normalized_coords_bihw + tf.squeeze(jacobian_biohw[:,:,o,:,:]) * output_perturbation_st[s, t]

    normalized_coords_bihws = tf.expand_dims(normalized_coords_bihw, axis = -1)
    normalized_coords_bihwst = tf.expand_dims(normalized_coords_bihws, axis = -1)

    input_perturbations_bihwst = tf.einsum('biohw,sto->bihwst', jacobian_biohw, output_perturbation_sto)
    perturbed_normalized_coords_bihwst = normalized_coords_bihwst + input_perturbations_bihwst

    # so then, i'd just want to sum over st
    # but the idea would be to keep that going all the way to the gather_nd() inside get_pixel_value(), so that the gatherings would be localized, for cache coherence
    # however, the lame way would be

    output_bhwc = tf.zeros(shape = (b, oh, ow, c), dtype = tf.float32)
    for s in tf.range(0, supersample):
        for t in tf.range(0, supersample):
            perturbed_normalized_coords_bihw = perturbed_normalized_coords_bihwst[:, :, :, :, s, t]
            output_bhwc += drizzle_sample_centers(input_bhwc, perturbed_normalized_coords_bihw, pixfrac)
    output_bhwc /= tf.cast(tf.square(supersample), dtype=output_bhwc.dtype)
    return output_bhwc
    

def drizzle_sample_centers(input_bhwc, normalized_coords_bihw, pixfrac):

    ih = tf.shape(input_bhwc)[1]
    iw = tf.shape(input_bhwc)[2]
    size_bihw = tf.reshape(tf.stack([iw, ih]), (1, 2, 1, 1))
    max_bihw = size_bihw - 1

    # convention is that the coordinate is the upper-left of the pixel
    pixel_coords_bihw = 0.5 * ((normalized_coords_bihw + 1.0) * tf.cast(size_bihw, 'float32'))

    floor_coords_bihw = tf.floor(pixel_coords_bihw)

    int_coords_bihw = tf.cast(floor_coords_bihw, 'int32')

    # these we can use to do the lookup
    zero = tf.zeros([], dtype='int32')    
    clipped_int_coords_bihw = tf.clip_by_value(int_coords_bihw, zero, max_bihw)

    within_input_bihw = tf.cast(tf.logical_and(tf.greater_equal(int_coords_bihw, zero), tf.greater_equal(max_bihw, int_coords_bihw)), dtype = tf.float32)

    within_input_bhw = tf.reduce_prod(within_input_bihw, axis=1)

    # coordinates within the pixel..
    subpixel_coords_bihw = pixel_coords_bihw - floor_coords_bihw

    dist_from_center_bihw = tf.abs(subpixel_coords_bihw - 0.5)

    within_pixfrac_bihw = tf.cast(tf.greater(pixfrac * 0.5, dist_from_center_bihw), dtype = tf.float32)

    within_pixfrac_bhw = tf.reduce_prod(within_pixfrac_bihw, axis=1)

    weight_bhw = within_input_bhw * within_pixfrac_bhw
    
    # same weights for all channels... hmm... makes me think of cool debayering stuff that could be done...
    weight_bhwc = tf.expand_dims(weight_bhw, axis = 3)
    
    pixel_values_bhwc = get_pixel_value(input_bhwc, clipped_int_coords_bihw[:, 0, :, :], clipped_int_coords_bihw[:, 1, :, :])

    output_bhwc = pixel_values_bhwc * weight_bhwc

    return output_bhwc