import os
import os.path
import tensorez.align as align
import tensorez.util as util
import tensorflow as tf

from tensorez.util import *

output_path = os.path.join('output', 'align_test')

file_glob = os.path.join('data', '2022-01-11_jwst', '2022-01-11-0941_0-CapObj.SER')
file_glob_darks = os.path.join('data', '2022-01-11_jwst', '2022-01-11-1027_1-CapObj.SER')

# loop below really doesn't need the full average, but we do need the shape and image count, so may as well make a pass
average_image_bhwc, image_count = load_average_image(file_glob)

image_shape = average_image_bhwc.shape
image_hw = image_shape.as_list()[-3:-1]

if file_glob_darks is not None:
    dark_image, dark_image_count = load_average_image(file_glob_darks)
    average_image_bhwc = average_image_bhwc - dark_image
        
mask_image = align.generate_mask_image(tf.shape(average_image_bhwc))
util.write_image(mask_image, os.path.join(output_path, 'mask_image.png'))

util.write_image(average_image_bhwc, os.path.join(output_path, 'initial_average_image.png'), normalize = True)

translation_only = False
learning_rate = .001
max_update_pixels = 0.9
max_steps = 30
passes_per_lod = 4
lod_factor = 1.2
min_size = 64

# could compute different values for width and height, and even for rotation, but KISS:
max_dimension = max(image_hw[-2], image_hw[-1])
max_update_magnitude = max_update_pixels / max_dimension

num_lods = 0
while pow(lod_factor, -num_lods) * max_dimension > min_size:
    num_lods += 1
print(f"num_lods: {num_lods}")


max_update_rot_factor = max_update_magnitude * .5
if translation_only:
    max_update_rot_factor = 0

max_update = tf.constant([max_update_magnitude, max_update_magnitude, max_update_rot_factor])

max_gradient = max_update / learning_rate

downsample_steps = max(0, int(math.log2(max_dimension) - 4))

#identity_alignment_transform = tf.eye(2, 3)

# dx, dy, theta
identity_alignment_transform = tf.zeros(3)

alignment_transforms = tf.Variable(tf.tile(tf.expand_dims(identity_alignment_transform, axis=0), multiples=(image_count, 1)))


#todo: really it should be "align this image with the average so far in this pass", rather than the current embarassingly-parallel approach...
#that has some annoying asymmetry to it (maybe helped with exponential moving average?) if there's individual best frames, but makes the problem easier
# also todo: instead of this downsampilng business, just do a big fft blur, perhaps with a gaussian or exponential distribution...

def read_average_image_with_alignment_transforms(file_glob, alignment_transforms, image_shape, dark_image):
    target_image = tf.zeros(image_shape)
    image_index = -1
    for filename in glob.glob(file_glob):
        for image_bhwc, frame_index in ImageSequenceReader(filename, to_float = True, demosaic = True):
            image_index += 1

            # must be in the loop
            if dark_image is not None:
                image_bhwc -= dark_image

            alignment_transform = alignment_transforms[image_index, :]
            transformed_image_bhwc = align.transform_image(image_bhwc, alignment_transform)
            target_image = target_image + transformed_image_bhwc
    target_image = target_image / image_count
    return target_image

for lod in range(num_lods - 1, -1, -1):

    downsample_factor = pow(lod_factor, lod)

    max_gradient_for_lod = max_gradient * downsample_factor
    max_update_magnitude_for_lod = max_update_magnitude * downsample_factor
    
    downsample_hw = [int(image_hw[0] / downsample_factor), int(image_hw[1] / downsample_factor)]
    
    for pass_index in range(passes_per_lod):
        
        # recompute the average image, but with the transforms applied
        target_image = read_average_image_with_alignment_transforms(file_glob, alignment_transforms, image_shape, dark_image)

        # now we downsample the target (should be equivalent to doing it in the loop above because linearity)
        if downsample_factor != 1:
            target_image = tf.image.resize(target_image, downsample_hw, antialias=True)
            downsampled_mask_image = tf.image.resize(mask_image, downsample_hw, antialias=True)
        else:
            downsampled_mask_image = mask_image

        util.write_image(target_image, os.path.join(output_path, 'target_image.png'))

        # now try to align everything to the new target
        image_index = -1
        for filename in glob.glob(file_glob):
            for image_bhwc, frame_index in ImageSequenceReader(filename, to_float = True, demosaic = True):
                image_index += 1

                if dark_image is not None:
                    image_bhwc = image_bhwc - dark_image

                if downsample_factor != 1:
                    image_bhwc = tf.image.resize(image_bhwc, downsample_hw, antialias=True)

                alignment_transform = tf.Variable(alignment_transforms[image_index, :])

                best_alignment_transform = tf.Variable(alignment_transform)

                #optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = 0.9, clipnorm = max_update_magnitude / learning_rate)
                optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, clipnorm = max_update_magnitude_for_lod / learning_rate)
                #optimizer = tf.keras.optimizers.Adagrad(learning_rate = learning_rate, clipnorm = max_update_magnitude / learning_rate)
                # hmm, this one didn't converge...
                #optimizer = tf.keras.optimizers.Adamax(learning_rate = learning_rate, clipnorm = max_update_magnitude / learning_rate)
                #optimizer = tf.keras.optimizers.RMSprop(learning_rate = learning_rate, clipnorm = max_update_magnitude / learning_rate)

                variable_list = [alignment_transform]

                steps = 0
                lowest_loss = float('inf')
                steps_since_lowest_loss = 0
                while True:    
                    #print(f'\n\nalignment_transform:\n{alignment_transform}')

                    with tf.GradientTape() as tape:
                        tape.watch(variable_list)

                        alignment_guess_image = align.transform_image(image_bhwc, alignment_transform)
                        guess_loss = align.alignment_loss(alignment_guess_image, target_image, downsampled_mask_image)
                        
                        #print(f'guess_loss: {guess_loss}')
                        
                        #util.write_image(alignment_guess_image, os.path.join(output_path, 'alignment_guess_image.png'))

                    # hand training
                    #gradient = tape.gradient(guess_loss, alignment_transform)
                    #transform_update = gradient * learning_rate
                    #transform_update = tf.clip_by_value(transform_update, -max_update, max_update)
                    #alignment_transform.assign(alignment_transform - transform_update)
                    #print(f'gradient: {gradient}')

                    gradients = tape.gradient(guess_loss, variable_list)
                    gradients[0] = tf.clip_by_value(gradients[0], -max_gradient_for_lod, max_gradient_for_lod)
                    optimizer.apply_gradients(zip(gradients, variable_list))

                    # hmm, need to orthonormalize?  otherwise, if the image is too dark, it may help for it to bring in dark pixels from the edge...

                    steps += 1
                    if guess_loss < lowest_loss:
                        lowest_loss = guess_loss
                        steps_since_lowest_loss = 0
                        best_alignment_transform.assign(alignment_transform) 
                    else:
                        steps_since_lowest_loss += 1

                    if steps_since_lowest_loss > 5 or steps > max_steps:
                        break
                
                alignment_transforms[image_index, :].assign(best_alignment_transform)

            
final_average_image_bhwc = read_average_image_with_alignment_transforms(file_glob, alignment_transforms, image_shape, dark_image)
util.write_image(final_average_image_bhwc, os.path.join(output_path, 'final_average_image.png'))

image_index = -1
for filename in glob.glob(file_glob):
    for image_bhwc, frame_index in ImageSequenceReader(filename, to_float = True, demosaic = True):
        image_index += 1

        # must be in the loop
        if dark_image is not None:
            image_bhwc -= dark_image

        alignment_transform = alignment_transforms[image_index, :]
        transformed_image_bhwc = align.transform_image(image_bhwc, alignment_transform)
        write_image(transformed_image_bhwc, os.path.join(output_path, "aligned_{:08d}.png".format(image_index)))