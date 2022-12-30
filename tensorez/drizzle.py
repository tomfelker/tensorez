"""
drizzle.py - a Python / Tensorflow implementation of the Drizzle algorithm
by Tom Felker
code based on and inspired by Kevin Zakka's implementation of Spatial Transformer Network (MIT license)
    - https://github.com/kevinzakka/spatial-transformer-network 
    - itself based on 'Spatial Transformer Networks', Jaderberg et. al,
        - https://arxiv.org/abs/1506.02025
and with reference to the Drizzle paper by Fruchter, A. S.; Hook, R. N.:
    - https://arxiv.org/pdf/astro-ph/9808087.pdf
    - DOI 10.1086/338393

My convention is for the names of tensors to end with the interpretations of their indices, in abbreviated form.
Those abbreviations mean:
    b: batch (for handling multiple images at once)
    h: height (0 <= y < height)
    w: width (0 <= x < width)
    c: channel (rgb)
    i: coordinate in input image (0 is x_i, 1 is y_i)
    o: coordinate in output image (0 is x_o, 1 is y_o)
    s: sample (for doing supersampling)

"""

import tensorflow as tf

@tf.function
def drizzle(input_image_bhwc, theta, upscale = 2, pixfrac = .5, out_shape_hw=None, inputspace_weights_bhwc = None, supersample = 4, jitter = True):
    """ Uses the Drizzle algorithm to transform and upscale images.
    
    Args:
        input_image_bhwc:
            The images to drizzle.

        theta:
            A 2d affine transform matrix for each image, which will be reshaped to [b, 2, 3].

            This has the same convention as Spatial Transformer Network.
            If output_coord is a column vector (in homogenous coordinates, so with a 1 at the bottom), then

            input_coord = theta * output_coord

            Both coordinates are normalized so that the image covers [-1, 1], so rotations will be about the center.

        upscale:
            How much bigger the output should be from the input.  (Don't simultaneously give out_shape_hw.)

            This is 's' in the DRIZZLE paper.

            Note that we're currently not scaling the image by s^2 as the paper does.  Doing so makes sense
                when using units of irradiance (power per area), but not for radiance (power per area per solid angle), which is the normal convention in computer graphics.
                Arguably we should scale the values (by the determinant of the jacobian) if theta doesn't preserve area, but we're not doing that yet.

        pixfrac:
            The ratio of the linear size of the drop to the input pixel.  Should be between 0 and 1.

            0 would be the equivalent of 'interlacing' (but don't go too close to zero, or you would need a lot of supersampling to avoid missing pixels)
            1 would be the equivalent of 'shift-and-add' (which seems to be what astronomy people call bilinear sampling)
          
        out_shape_hw:
            The height and width you'd like the output to be.  (if you're using this, pass upscale=None, and we will infer it.)

        inputspace_weights_bhwc:
            Weights to multply in, per input pixel.  (h and w refer to the input image dimensions.)

            This is where, for example, you could pass 0 for dead pixels or cosmic ray hits.

        supersample:
            How many samples to take in each direction per pixel.  (e.g., supersample=4 samples in a 4x4 grid, so 16 samples.)  Pass 1 to disable, don't pass 0.

            We aren't computing the overlaps analytically, because with distortion or weird transforms, there's no limit to how many input pixels could touch an output pixel.
            Instead, we take a grid of samples in the output pixel, and for each sample, determine whether it's within the 'drip', i.e. near the center, of the input pixel.

        jitter:
            If True, jitter the positions of the samples within the pixel.  This should help prevent aliasing / moire patterns from affecting the weights.

    Returns:
        A tuple (output_image_bhwc, output_weights_bhwc), where:

            output_image_bhwc is the resulting images.  Places where the input pixels fell have the corresponding values, other places have 0.
        
            output_weights_bhwc is the resulting weights.  Places where the input pixels fell have 1 (times any weights given), other places have 0.
    """

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

    if supersample != 1 or jitter is True:
        # TODO: This is doing it numerically, and computing the jacobian at every pixel.
        # that'll be helpful if we have per-pixel shifts (geometric distortion, local alignment, that sort of thing)
        # but it's actually completely pointless for affine transforms, since the valaues will actually be the same for each pixel.
        # so, in that case, we could save some memory and time by just giving a single value that can be broadcasted as needed.
        jacobian_biohw = jacobian_from_grid(grid_bihw)
    else:
        # not needed when we're not supersampling.
        jacobian_biohw = None

    output_image_bhwc, output_weights_bhwc = drizzle_sampler(input_image_bhwc, inputspace_weights_bhwc, grid_bihw, pixfrac, supersample, jitter, jacobian_biohw)

    return output_image_bhwc, output_weights_bhwc


def get_pixel_value(image_bhwc, xi_coords_bhws, yi_coords_bhws):
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
    shape = tf.shape(xi_coords_bhws)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    samples = shape[3]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1))
    b = tf.tile(batch_idx, (1, height, width, samples))

    indices = tf.stack([b, yi_coords_bhws, xi_coords_bhws], axis = -1)

    output_bhwsc = tf.gather_nd(image_bhwc, indices)

    output_bhwc = tf.reduce_mean(output_bhwsc, axis = -2)

    return output_bhwc


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

def drizzle_sampler(input_bhwc, inputspace_weights_bhwc, normalized_coords_bihw, pixfrac, supersample, jitter, jacobian_biohw):
    if supersample == 1 and jitter is False:
        normalized_coords_bihws = tf.expand_dims(normalized_coords_bihw, axis = -1)
        return drizzle_sample_centers(input_bhwc, normalized_coords_bihw, pixfrac)

    b = normalized_coords_bihw.shape[0]
    oh = normalized_coords_bihw.shape[2]
    ow = normalized_coords_bihw.shape[3]
    c = input_bhwc.shape[-1]

    # how much we perturb our output normalized coords
    # s and t are the x,y coordinates within the supersample grid
    ssx_s = tf.range(0, supersample, dtype = tf.float32)
    ssy_t = tf.range(0, supersample, dtype = tf.float32)

    output_perturbation_samples_sto = tf.stack(tf.meshgrid(ssx_s, ssy_t), axis = -1)

    # nobody else cares that our samples are in a grid (and for fancier sampling, they wouldn't be), so collapse that to one dimension,
    # and add extra dimensions that we'll need if jittering.  (When not, they'll broadcast.)
    output_perturbation_samples_bhwso = tf.reshape(output_perturbation_samples_sto, shape=(1, 1, 1, supersample * supersample, 2))

    if jitter:
        # un-broadcast since we need a jitter value for every pixel in every image
        output_perturbation_samples_bhwso = tf.tile(output_perturbation_samples_bhwso, multiples = (b, oh, ow, 1, 1))
        output_perturbation_samples_bhwso += tf.random.uniform(shape = output_perturbation_samples_bhwso.shape) - 0.5

    # convert from "samples" to "normalized output coordinates"
    output_size_bhwso = tf.cast(tf.reshape(tf.stack([ow, oh]), shape = (1, 1, 1, 1, 2)), dtype = tf.float32)
    output_perturbation_bhwso = ((output_perturbation_samples_bhwso + 0.5) / supersample - 0.5) / output_size_bhwso
    
    normalized_coords_bihws = tf.expand_dims(normalized_coords_bihw, axis = -1)

    # This is summing over o (the index of the output coordinate).  In other words, at each sample, we're perturbing the output coordinate
    # by given amounts along the output-x and output-y directions, and for each of those directions, we must multiply by the the jacobian
    # to figure out how much to perturb the input x and y coordinates.
    input_perturbations_bihws = tf.einsum('biohw,bhwso->bihws', jacobian_biohw, output_perturbation_bhwso)

    perturbed_normalized_coords_bihws = normalized_coords_bihws + input_perturbations_bihws

    return drizzle_sample_centers(input_bhwc, inputspace_weights_bhwc, perturbed_normalized_coords_bihws, pixfrac)
    

def drizzle_sample_centers(input_bhwc, inputspace_weights_bhwc, normalized_coords_bihws, pixfrac):

    ih = tf.shape(input_bhwc)[1]
    iw = tf.shape(input_bhwc)[2]
    size_bihws = tf.reshape(tf.stack([iw, ih]), (1, 2, 1, 1, 1))
    max_bihws = size_bihws - 1

    # convention is that the coordinate is the upper-left of the pixel
    pixel_coords_bihws = 0.5 * ((normalized_coords_bihws + 1.0) * tf.cast(size_bihws, 'float32'))

    floor_coords_bihws = tf.floor(pixel_coords_bihws)

    int_coords_bihws = tf.cast(floor_coords_bihws, 'int32')

    # these we can use to do the lookup
    zero = tf.zeros([], dtype='int32')    
    clipped_int_coords_bihws = tf.clip_by_value(int_coords_bihws, zero, max_bihws)

    within_input_bihws = tf.cast(tf.logical_and(tf.greater_equal(int_coords_bihws, zero), tf.greater_equal(max_bihws, int_coords_bihws)), dtype = tf.float32)

    within_input_bhws = tf.reduce_prod(within_input_bihws, axis=1)

    # coordinates within the pixel..
    subpixel_coords_bihws = pixel_coords_bihws - floor_coords_bihws

    dist_from_center_bihws = tf.abs(subpixel_coords_bihws - 0.5)

    within_pixfrac_bihws = tf.cast(tf.greater(pixfrac * 0.5, dist_from_center_bihws), dtype = tf.float32)

    within_pixfrac_bhws = tf.reduce_prod(within_pixfrac_bihws, axis=1)

    # geometric meaning 'the weight we get because of the geometry of how the drips map to the output pixels'
    geometric_weight_bhws = within_input_bhws * within_pixfrac_bhws

    # get_pixel_value() will reduce_mean along the s axis, we must do the same for weights.
    geometric_weight_bhw = tf.reduce_mean(geometric_weight_bhws, axis = -1)
    
    # same weights for all channels... hmm... makes me think of cool debayering stuff that could be done...
    geometric_weight_bhwc = tf.expand_dims(geometric_weight_bhw, axis = 3)
    
    pixel_values_bhwc = get_pixel_value(input_bhwc, clipped_int_coords_bihws[:, 0, :, :, :], clipped_int_coords_bihws[:, 1, :, :, :])    

    # it's necessary to push the weights lookup all the way down here, rather than a seperate pass, so we use the same jitter values.
    output_weight_bhwc = geometric_weight_bhwc
    if inputspace_weights_bhwc is not None:
        inputspace_weight_values_bhwc = get_pixel_value(inputspace_weights_bhwc, clipped_int_coords_bihws[:, 0, :, :, :], clipped_int_coords_bihws[:, 1, :, :, :])
        output_weight_bhwc *= inputspace_weight_values_bhwc

    output_value_bhwc = pixel_values_bhwc * output_weight_bhwc

    return output_value_bhwc, output_weight_bhwc