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

    # TODO: This is doing it numerically, and computing the jacobian at every pixel.
    # that'll be helpful if we have per-pixel shifts (geometric distortion, local alignment, that sort of thing)
    # but it's actually completely pointless for affine transforms, since the valaues will actually be the same for each pixel.
    # so, in that case, we could save some memory and time by just giving a single value that can be broadcasted as needed.
    jacobian_biohw = jacobian_from_grid(grid_bihw)

    if inputspace_weights_bhwc is None:
        inputspace_weights_bhwc = tf.ones_like(input_image_bhwc)
    else:
        input_image_bhwc *= inputspace_weights_bhwc
    
    output_image_bhwc = drizzle_sampler(input_image_bhwc, grid_bihw, jacobian_biohw, pixfrac, supersample)

    output_weights_bhwc = drizzle_sampler(inputspace_weights_bhwc, grid_bihw, jacobian_biohw, pixfrac, supersample)

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

    # coordinate in upper-left of the pixel:
    #x = (tf.range(0, out_w, dtype = tf.float32) * 2.0 / out_w) - 1.0
    #y = (tf.range(0, out_h, dtype = tf.float32) * 2.0 / out_h) - 1.0

    # coordinate in center of pixel:
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
    padded_grid_oxm1_bihw = tf.roll(padded_grid_bihw, shift = -1, axis = 0)
    padded_grid_oxp1_bihw = tf.roll(padded_grid_bihw, shift =  1, axis = 0)
    padded_grid_oym1_bihw = tf.roll(padded_grid_bihw, shift = -1, axis = 1)
    padded_grid_oyp1_bihw = tf.roll(padded_grid_bihw, shift =  1, axis = 1)

    # these are copies of valid, shifted in each direction.  Read oxm1 as "output x minus 1"
    padded_valid_oxm1_bihw = tf.roll(padded_valid_bihw, shift = -1, axis = 0)
    padded_valid_oxp1_bihw = tf.roll(padded_valid_bihw, shift =  1, axis = 0)
    padded_valid_oym1_bihw = tf.roll(padded_valid_bihw, shift = -1, axis = 1)
    padded_valid_oyp1_bihw = tf.roll(padded_valid_bihw, shift =  1, axis = 1)

    # the denominators account for how many valid coordinates we had (which should be 1 or 2)

    # read as "change in input coordinate i per change in output x coordinate"
    di_over_dxo_bihw = (padded_grid_oxp1_bihw - padded_grid_oxm1_bihw) / ((padded_valid_oxp1_bihw + padded_valid_oxm1_bihw) * out_w)
    # read as "change in input coordinate i per change in output y coordinate"
    di_over_dyo_bihw = (padded_grid_oyp1_bihw - padded_grid_oym1_bihw) / ((padded_valid_oyp1_bihw + padded_valid_oym1_bihw) * out_h)
    
    # could also have named this di_over_do_biohw, which you could read as "change in input coordinate i per change in output coordinate o"
    padded_jacobian_biohw = tf.stack([di_over_dxo_bihw, di_over_dyo_bihw], axis = 2)

    # unpad
    jacobian_biohw = padded_jacobian_biohw[:, :, :, 1 : out_h, 1 : out_w]

    return jacobian_biohw


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.

    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # hax todo find right way to modify this
    # was:
    #x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    #y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))
    # but should be:
    x = 0.5 * ((x + 1.0) * tf.cast(max_x, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # hax - also changing it so we only clip the indices, but not the coordinates
    # used in the delta calcuation below.  This seems to have the effect of changing the borders
    # from being black (and cutting off the lower-right edge with the identity transform)
    # to being clamp to nearest

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0_clipped = tf.clip_by_value(x0, zero, max_x)
    x1_clipped = tf.clip_by_value(x1, zero, max_x)
    y0_clipped = tf.clip_by_value(y0, zero, max_y)
    y1_clipped = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0_clipped, y0_clipped)
    Ib = get_pixel_value(img, x0_clipped, y1_clipped)
    Ic = get_pixel_value(img, x1_clipped, y0_clipped)
    Id = get_pixel_value(img, x1_clipped, y1_clipped)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out


def drizzle_sampler(input_bhwc, grid_bihw, jacobian_biohw, pixfrac, supersample):
    xi_coords_bhw = grid_bihw[:, 0, :, :]
    yi_coords_bhw = grid_bihw[:, 1, :, :]

    # todo: supersample, needs this - huh, i guess the other way doesn't?
    dxi_over_dxo_bhw = jacobian_biohw[:, 0, 0, :, :]
    dxi_over_dyo_bhw = jacobian_biohw[:, 0, 1, :, :]
    dyi_over_dxo_bhw = jacobian_biohw[:, 1, 0, :, :]
    dyi_over_dyo_bhw = jacobian_biohw[:, 1, 1, :, :]

    ih = tf.shape(input_bhwc)[1]
    iw = tf.shape(input_bhwc)[2]
    max_yi = tf.cast(ih - 1, 'int32')
    max_xi = tf.cast(iw - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # convention is that the coordinate is the upper-left of the pixel
    xi_pixel_coords_bhw = 0.5 * ((xi_coords_bhw + 1.0) * tf.cast(ih, 'float32'))
    yi_pixel_coords_bhw = 0.5 * ((yi_coords_bhw + 1.0) * tf.cast(iw, 'float32'))

    xi_floor_coords_bhw = tf.floor(xi_pixel_coords_bhw)
    yi_floor_coords_bhw = tf.floor(yi_pixel_coords_bhw)

    xi_int_coords_bhw = tf.cast(xi_floor_coords_bhw, 'int32')
    yi_int_coords_bhw = tf.cast(yi_floor_coords_bhw, 'int32')

    # these we can use to do the lookup    
    xi_clipped_int_coords_bhw = tf.clip_by_value(xi_int_coords_bhw, zero, max_xi)
    yi_clipped_int_coords_bhw = tf.clip_by_value(yi_int_coords_bhw, zero, max_yi)

    xi_within_input_bhw = tf.logical_and(tf.greater_equal(xi_int_coords_bhw, zero), tf.greater_equal(max_xi, xi_int_coords_bhw))
    yi_within_input_bhw = tf.logical_and(tf.greater_equal(yi_int_coords_bhw, zero), tf.greater_equal(max_yi, yi_int_coords_bhw))
    within_input_bhw = tf.logical_and(xi_within_input_bhw, yi_within_input_bhw)

    # coordinates within the pixel..
    xi_subpixel_coords_bhw = xi_pixel_coords_bhw - xi_floor_coords_bhw
    yi_subpixel_coords_bhw = yi_pixel_coords_bhw - yi_floor_coords_bhw

    xi_dist_from_center_bhw = tf.abs(xi_subpixel_coords_bhw - 0.5)
    yi_dist_from_center_bhw = tf.abs(yi_subpixel_coords_bhw - 0.5)

    xi_within_pixfrac_bhw = tf.greater(pixfrac * 0.5, xi_dist_from_center_bhw)
    yi_within_pixfrac_bhw = tf.greater(pixfrac * 0.5, yi_dist_from_center_bhw)
    within_pixfrac_bhw = tf.logical_and(xi_within_pixfrac_bhw, yi_within_pixfrac_bhw)

    weight_bhw = tf.cast(tf.logical_and(within_input_bhw, within_pixfrac_bhw), dtype = input_bhwc.dtype)
    
    # same weights for all channels... hmm... makes me think of cool debayering stuff that could be done...
    weight_bhwc = tf.expand_dims(weight_bhw, axis = 3)
    
    pixel_values_bhwc = get_pixel_value(input_bhwc, xi_clipped_int_coords_bhw, yi_clipped_int_coords_bhw)

    output_bhwc = pixel_values_bhwc * weight_bhwc

    return output_bhwc