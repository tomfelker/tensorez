import numpy as np
import tensorflow as tf
import os
import glob
from tensorstax_util import *
from tensorstax_model import *

model_adc = True
psf_size = 128
batch_size = 40

psf_training_steps = 10
psf_learning_rate = .001

image_training_steps = 10
image_learning_rate = .001

overall_training_steps = 100

#data_path = os.path.join('data', 'saturn_bright_mvi_6902')
data_path = os.path.join('data', 'jupiter_mvi_6906')
#data_path = os.path.join('obd', 'data','epsilon_lyrae')

file_glob = os.path.join(data_path, '????????.png')

output_dir = os.path.join("output", "latest")


tf.enable_eager_execution()

# for now, just a single batch
print("Reading input images...")
num_images = 0
images = []
for filename in glob.glob(file_glob):
    images.append(read_image(filename, to_float = not model_adc))
    num_images += 1
    if num_images == batch_size:
        break
images = tf.stack(images, axis = 0)


print("Instantiating model.")
model = TensorstaxModel(psf_size = psf_size, model_adc = model_adc)

print("Beginning training....")

image_optimizer = tf.train.AdamOptimizer(image_learning_rate)

psf_optimizer = tf.train.AdamOptimizer(psf_learning_rate)

for overall_training_step in range(0, overall_training_steps):

    # init these inside the loop, so their momenta don't persist across updates
    


    for psf_training_step in range(0, psf_training_steps):    
        with tf.GradientTape() as psf_tape:
            model(images)

        print("Overall step {}, psf step {}: loss {}".format(overall_training_step, psf_training_step, sum(model.losses)))

        grads = psf_tape.gradient(model.losses, model.psf_training_vars)
        psf_optimizer.apply_gradients(zip(grads, model.psf_training_vars))

        model.apply_psf_physicality_constraints()

        psf_examples = model.get_psf_examples()
        write_image(psf_examples, os.path.join(output_dir, "psf_examples_latest_{}.png".format(psf_training_step)))        

    psf_examples = model.get_psf_examples()
    write_image(psf_examples, os.path.join(output_dir, "psf_examples_latest.png"))
    write_image(psf_examples, os.path.join(output_dir, "psf_examples_{}.png".format(overall_training_step)))

    for image_training_step in range(0, image_training_steps):
        with tf.GradientTape() as image_tape:
            model(images)

        grads = image_tape.gradient(model.losses, model.image_training_vars)
        image_optimizer.apply_gradients(zip(grads, model.image_training_vars))

        print("Overall step {}, image step {}: loss {}".format(overall_training_step, image_training_step, sum(model.losses)))

        model.apply_image_physicality_constraints()

    estimated_image = model.estimated_image
    write_image(estimated_image, os.path.join(output_dir, "estimated_image_latest.png"))    
    write_image(estimated_image, os.path.join(output_dir, "estimated_image_{}.png".format(overall_training_step)))

    estimated_image_centered = center_image_per_channel(estimated_image)
    write_image(estimated_image_centered, os.path.join(output_dir, "estimated_image_centered_latest.png"))    
    write_image(estimated_image_centered, os.path.join(output_dir, "estimated_image_centered_{}.png".format(overall_training_step)))

    model.print_adc_stats()
    

print("Cool");

    
