"""
Fourier transforms are scary...

Idea here is to do lucky imaging, but in a more data-efficient way.

For lucky imaging, you're taking the "best" images, something like 10% of them - but you're doing this on the basis of the whole image at once,
and also shifting on the basis of the whole image at once.

So how about this:

Forward model of the image:  the true image
a distortion field
dot product the observed image
as an estimate of how lucky this patch of this iamge is

can we unify this with panning over everything, and removal of the telescope's diffraction?

... for each frame:  take true image, do a pan/crop to get to the sensor, convolve by system's PSF, and that's your expected observation
... just one PSF for the whole sequence, i.e., only the unchanging telescope, not caring about atmospheric PSF crap
... then, do the distortion field thing (or psf field thing)
... then, do your dot product to determine "luckiness" - in parts of the image where you're lucky, it tells you more about the true image

"""