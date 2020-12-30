import os
import os.path
import tensorez.align as align
import tensorez.util as util
import tensorflow as tf

output_path = os.path.join('output', 'align_test')
target_image_path = os.path.join('data', 'align_test', 'target.png')
unaligned_image_path = os.path.join('data', 'align_test', 'translated.png')

target_image = util.read_image(target_image_path)
unaligned_image = util.read_image(unaligned_image_path)

mask_image = align.generate_mask_image(tf.shape(target_image))
util.write_image(mask_image, os.path.join(output_path, 'mask_image.png'))

util.write_image(target_image, os.path.join(output_path, 'target_image.png'))

# okay here's what we want to try, just riffing for now...


translation_only = True
learning_rate = .001
max_update_pixels = 0.9

# could compute different values for width and height, and even for rotation, but KISS:
max_update_magnitude = max_update_pixels / max(target_image.shape[-2].value, target_image.shape[-3].value)

max_update = tf.constant([
    [0.0 if translation_only else max_update_magnitude, 0.0 if translation_only else max_update_magnitude, max_update_magnitude],
    [0.0 if translation_only else max_update_magnitude, 0.0 if translation_only else max_update_magnitude, max_update_magnitude]
])

max_gradient = max_update / learning_rate

alignment_transform = tf.Variable(tf.eye(2, 3))


rotate_transform = tf.constant([
    [1.0, 0.0, 0.5],
    [0.0, 1.0, 0.2],
])
identity_test_image = align.transform_image(target_image, rotate_transform)
util.write_image(identity_test_image, os.path.join(output_path, 'identity_test_image.png'))




best_alignment_transform = tf.Variable(alignment_transform)

#optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = 0.9, clipnorm = max_update_magnitude / learning_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, clipnorm = max_update_magnitude / learning_rate)
#optimizer = tf.keras.optimizers.Adagrad(learning_rate = learning_rate, clipnorm = max_update_magnitude / learning_rate)
# hmm, this one didn't converge...
#optimizer = tf.keras.optimizers.Adamax(learning_rate = learning_rate, clipnorm = max_update_magnitude / learning_rate)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate = learning_rate, clipnorm = max_update_magnitude / learning_rate)

variable_list = [alignment_transform]

steps = 0
lowest_loss = float('inf')
steps_since_lowest_loss = 0
while True:    
    print(f'\n\nalignment_transform:\n{alignment_transform}')

    with tf.GradientTape() as tape:
        tape.watch(variable_list)

        alignment_guess_image = align.transform_image(unaligned_image, alignment_transform)
        guess_loss = align.alignment_loss(alignment_guess_image, target_image, mask_image)
        
        print(f'guess_loss: {guess_loss}')
        
        util.write_image(alignment_guess_image, os.path.join(output_path, 'alignment_guess_image.png'))

    # hand training
    #gradient = tape.gradient(guess_loss, alignment_transform)
    #transform_update = gradient * learning_rate
    #transform_update = tf.clip_by_value(transform_update, -max_update, max_update)
    #alignment_transform.assign(alignment_transform - transform_update)
    #print(f'gradient: {gradient}')

    gradients = tape.gradient(guess_loss, variable_list)
    gradients[0] = tf.clip_by_value(gradients[0], -max_gradient, max_gradient)
    optimizer.apply_gradients(zip(gradients, variable_list))

    steps += 1
    if guess_loss < lowest_loss:
        lowest_loss = guess_loss
        steps_since_lowest_loss = 0
        best_alignment_transform.assign(alignment_transform) 
    else:
        steps_since_lowest_loss += 1

    if steps_since_lowest_loss > 20: #steps * .5:
        break

print(f'finished after {steps} with loss {lowest_loss}')