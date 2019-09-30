# TensoRez

Enhance your images with the magic of reverse-mode automatic differentiation!

The goal is to learn what was present among the fixed stars, given only an imperfect video distorted by the dynamic atmosphere.  Other software can do this, but TensoRez:
- doesn't require much manual effort or input
- has only a few magic numbers
- doesn't let humans overfit the data
- can take advantage of cutting edge compute resources
- is open source, and can be improved by all

It has some disadvantages:
- It's unfinished, and has no GUI.
- It will take hours to run, even with a state-of-the-art GPU.
- You'll need to install tensorflow-gpu, tensorflow-graphics, pillow, pyraw, et al.
- You'll need to open a text editor.

## Design

The idea is, build a simulation of how the atmosphere, the telescope, and the camera distort the view of the night sky.  Inintialize that simulation with your best guess of what's in the sky.  Let the simulation tell you what you would see, and compare it with what you actually saw.  Then update the parameters of the simulation, including its estimate of the true sky, to reduce the error in that comparison.

The model assumes that there is a constant night sky, but each frame of your video has been corrupted by the atmosphere.  This is modeled as a per-frame point-spread function, which is convolved over the estimate of the night sky image, to yield the estimate of what each frame of your video should have been.  (The frames will be further modifed by Bayer filtering, noise, and ADC transfer functions.)  If the model is correct, then hopefully when its prediction error is minimized, its estimate of the night sky will match the truth.

## Installation

Not much help here... install a recent Python and a recent TensorFlow. (I'm being cutting edge, and using eager mode).  You will want to use a GPU, unless you want to run things for days at a time.  Then Python -m PIP install the following:
- tensorflow-graphics
- pillow
- pyraw

Then you should be able to just grab this repo, and look at tensorez.py.  It has a bunch of settings you will want to edit, and you can point it at your data.  Then run it, and watch the pictures appear.

## Future directions

- Synthetic data.  By using the model to produce corrupted images of a known sky, and then trying to rediscover the model's input, we can have an objective metric of quality, determine the importance of various effects, and better tune the training process.  Plus this way I don't have to go outside where it might be cold.

- Bayer filter processing.  In theory, by modeling the camera's bayer filter, and looking at the raw, un-demosaiced data, we can recover information that would otherwise be lost.  My DSLR has an antialiasing filter that may ruin this, but it may be possible with data from raw CCD sensors.

- Minibatch processing.  Currently all images are loaded at once, and combined with super-resolution, large PSFs, and gradients, this limits processing to a few hundred frames.  By loading and saving the per-frame data, we could process larger datasets.

- Per-frame transforms.  Currently we simply center the images (by center-of-mass) before processing, but if the truth is rotating (such as a video of a satellite pass), each frame should have at least an affine transform.  It would also be nice to have autocorrelation-based alignment.  Spatial-transformer-network seems useful here.


