Tensorez
===
Using modern tensor libraries to enhance the resolution of your telescope.

The idea here is to make an easy to use, principled, opinionated, fast, and modern application to help with alignment, stacking, lucky imaging, deconvolution, and various other techniques for processing videos from a telescope, especially of planets and satellites.

Installing
==
TODO

Using
==
TODO

Architecture
==
There's a command-line python script, written with PyTorch, that handles all the actual image processing.  It uses 'recipe' files, which specify all the input filenames, and options for all the processing.  Then, there's a GUI, which basically provides a nice way to edit these recipes, and to run the CLI.

See DESIGN_CONTRACT.md for more on the details.

History
==
This project started some years ago, and at the time was written in Tensorflow.  The original idea was to build a forward model of a true image, being croupted by a series of PSFs, generating observed images - and then to fit this model using gradient descent to the actual observations, and then to extract the true image, which would be among the weights.  It didn't work very well, and while it could have been improved by factoring the problem to avoid directly solving for the true image, the logical endpoint has already been reached by others, and implemented in torchmfbd, so we can just use that as one step in our pipeline.

Then I implemented a few other random ideas:
* Online blind deconvolution, basically a tensorflow implementation of that paper's matlab code
* Some traditional lucky imaging stuff
* Some "local" lucky imaging, where we make a luckiness extimate per pixel (not just per frame), so it can work well on extended (e.g., lunar) images.
* Alignment code, using backprop to precisely align images
* "Local align", which aligns images using a flow field - also great for lunar images.

All of these ideas were just standalone python scripts, using some of the libary code developed here.  And in particular, a pipeline for loading the files and doing some preprocessing, like dark frame subtraction and computing per-pixel variance.

But now it's 2026, and vibe coding is a thing, and so now it's easy to rewrite this in PyTorch, with a cleaner 'recipe' architecture rather than hardcoding filenames at the top of the scripts, and even with a fancy UI, so that eventually it can be accessible to hobbyists who aren't comforable coding.