TensoRez
==========

Idea: Use the power of GPUs and Tensorflow and python and github, to write some software for planetary imaging that
- doesn't require manual picking of tracking points, who has time for that?
- isn't finicky, with few magic numbers
- doesn't let people tweak the result so much that it becomes more art than science
- is open source

So, we take a video of a planet, or the Moon, or a satellite, or whatever, through a telescope.

The object is very far away, but the light from it hits the Earth's atmosphere, and then gets diffracted around by my sadly finite-aperture telescope.

So we get the true image, convolved with the point-spread function (PSF) of the atmosphere (which changes quickly in time, and across the image), convolved with the PSF of the telescope (varying slowly in time, and not as much across the image).

So basic idea, keep a guess of the true image, and "train" all those PSFs using gradient descent.  Which luckily, is structurally identical to ConvNets, which is basically the main thing TensorFlow is tuned for.

This turns out to be similar to:

https://en.wikipedia.org/wiki/Richardson–Lucy_deconvolution
	- which by itself is a way of inverting a known convolution, but modified like this:
	- https://pdfs.semanticscholar.org/9e3f/a71e22caf358dbe873e9649f08c205d0c0c0.pdf
		- seems to be a method for 'blind deconvolution', i.e., also guessing the convolution kernel
		- paper written in 1995 with 128x128 images and kernels, but only one kernel per image
		
So I may just be basically implementing that paper, not sure if I even need the magic of automatic reverse mode automatic differentiation...
	main innovations:
		- using lots of varying convolutions across the image, and across time (perhaps regularizing on how quickly they vary)
		- layering in one that doesn't vary over time (which I hope will learn the telescope's PSFs, and let the others subsume the atmosphere's)
			- not sure if this separation is pure, or maybe it could be only with some complex numbers in between to let it capture phase?
			
			
decoding this paper
https://pdfs.semanticscholar.org/9e3f/a71e22caf358dbe873e9649f08c205d0c0c0.pdf
	key:
		P(x) == f(x) == "object distribution" = object
		P(y|x) == g(y, x) == "the PSF centered at x" = 
		P(y) == c(y) == "degraded image or convolution" = measured(y)
		
	equation 3:
		next_object_guess(x) = ((measured(x) / (object_guess(x) conv psf(x))) conv psf(-x)) * object_guess(x)
		
			- take our measurement, divided by our predicted measurement (our object guess convolved with our known PSF), to get a thing that should be 1, but will vary above and below 1 as the measurement is brighter or dimmer
			- convolve that with the mirrored PSF (hmm, why?)
			- and multiply with our object guess, to get the new improved object guess
		