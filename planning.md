## Local alignment:
To do good moon or extended object lucky imaging, need local alignment.
- can modify STN to allow per-pixel alignment tweaks, and learn them similar to align.py
    - started doing this in local_align.py
    - would still want to cache (as it's slow), but not in one big array (as it wouldn't fit in memory)
        - so need to set up some per-frame caching thing (heh, and buy an SSD)

## Drizzle:
- there is existing ancient C code (astropy.drizzle -> drizzlepac)
- seems like it'd be possible to modify STN to do drizzle
    - output would need to include a weight mask (1 for fully-covered pixels, 0 for not covered, and then after drizzling several images you'd divide by that)
    - bilinear_sample(), instead of that, the output would only read the input if it's within pixfrac of the center of the input pixel
    - grid generator should also include Jacobians (necessary even to do above correctly)
        - i.e., if output[ox,oy] should read input[ix, iy], need to know ∂ix/∂ox, ∂iy/∂ox, ∂ix/∂oy, ∂iy/∂oy
    - in lieu of being clever about pixel intersection area, just do a bunch of supersampling - i.e., do the above at a 4x4 grid over the output pixel instead of just once at the center
        - even lazier but equivalently - just drizzle to a much larger image, like 8x rather than 2x, and then downsample.  wastes memory vs above though.
    - okay, did this!  only problems:
        - fairly memory intensive
        - fairly slow

## Local lucky
- I don't trust my local statistics anymore...  it does seem that they'd get a double edge
    - use magnitude of derivatives of lowpassed image?  (Laplacian?)
        - why does mathy stuff say Laplacian is divergence of gradient, but imagey stuff treats it just as the gradient?
        - how to compute?
    - also annoying that i'm not using agreement with the known image, seems it'd help prevent problems
    - new theory:
        - lowpass of average image, lowpass of frame, mse - need low
            - this lowpass amount is related to seeing - means "frame matches image, to within the limit of our seeing"
            - should exclude any frequencies not present in the average image
            - this would help with not having local alignment
                - we'd discard areas that are sharp but misaligned - not ideal, but better than using them
        - gentle lowpass of frame (just to avoid pixel noise), magnitude of Laplacian, big lowpass
            - represents our luckiness
            - 'gentle lowpass' just to get rid of pixel noise (maybe also incorporate dark frame variance here, to unluckify areas around noisy pixels?)
            - big lowpass related to isoplanatic patch
    - this paper from times immemorial has some ideas for defining 'sharpness': https://apps.dtic.mil/sti/pdfs/AD0786157.pdf "REAL-T1ME CORRECTION OF ATMOSPHERICALLY DEGRADED TELESCOPE IMAGES THROUGH IMAGE SHARPENING" by Richard A. Muller, Andrew Buffington"

        - S_1:  reduce_sum(image*image)
            - lol wut, this works?  wtf, i guess that kinda makes sense...
            - idea being that any blurring will spread out the peaks, so will at best preserve reduce_sum(image), but knock down the sum of squares
            - seems easy to localize, just square the image and then blur it, and rely on the per-pixel normalization of luckiness to handle stuff
            - would probably need to normalize the images first, so we don't ignore frames with lost intensity for whatever reason
        - S_2: brightness of brightest pixel... meh
        - S_3: reduce_sum(image * true_image)
            - lol, if we knew true_image, why bother? though for stars, you can kinda assume true_image is Dirac delta
            - but they claim it works even if we're quite wrong about true_image
            - maybe that means it'll work for true_image = avg of all frames?
            - sounds like maximizing a dot product, that tracks
            - again, try blur(image*avg_image)
        - S_4
            - wish I knew what they meant here - what's m and n?  (or in appendix, cursive ℓ and m)
        - S_
- If I get drizzle working, some interesting questions about how to incorporate it...
    - some part of 'luckiness' should be applied to the raw image (pre drizzle, pre align (not doing that currently)
    - the 'agreement' stuff, ultimately also part of luckiness, in terms of average image (so, post-alignment, output space)
    - the final weight / sum stuff needs to be in output space

- frequency bands is working well - but, noticing some aliasing stuff
    - would be better to do any processing of the original image (bandpass filtering mostly) before aligning it, to avoid aliasing messing with stuff

## Bayer
- currently, I'm mostly demosasicing manually at the start, and then ignoring it from then on
- recently added support for Bayer in the local lucky image drizzling process (or non-drizzle)
- but!  we're using 3x the memory we really need, as demosaicing makes up a lot of numbers that aren't actually there
- Bayer Subchannels:
    - my idea is, don't have a 3-channel image at the full sensor resolution (two thirds of whose data is made up),
    - instead, have a 4-channel image at half the sensor resolution (no made up data).  The channels would correspond to the 4 squares of the bayer pattern.
        - Pros:
        - Cons:
            - need to remember the bayer pattern when finally saving the image out
            - the channels are each misaligned - no issue when shifting, but it will make rotations and other transformations slightly incorrect
                - could possibly fix with very fancy math
            - we throw out some of the benefit of two green channels... if there is a 1-px sharp green edge (oriented +45 degrees), each of the two green
            channels won't be able to tell that it isn't a less sharp 2px edge.