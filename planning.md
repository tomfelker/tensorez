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
- If I get drizzle working, some interesting questions about how to incorporate it...
    - some part of 'luckiness' should be applied to the raw image (pre drizzle, pre align (not doing that currently)
    - the 'agreement' stuff, ultimately also part of luckiness, in terms of average image (so, post-alignment, output space)
    - the final weight / sum stuff needs to be in output space
