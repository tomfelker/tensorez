from tensorez.util import align_by_center_of_mass, read_image
import tensorez.align as align
from tensorez.util import *
import tensorflow as tf
import hashlib
import os.path
import numpy as np

class Observation:
    def __init__(self, lights, darks = None, align_by_center_of_mass = False, crop = None, crop_align = 0, crop_before_align = False):
        self.lights = lights
        self.darks = darks
        self.align_by_center_of_mass = align_by_center_of_mass
        self.crop = crop
        self.crop_align = crop_align
        self.crop_before_align = crop_before_align

        self.dark_image = None
        self.alignment_transforms = None

    def load_or_create_dark_image(self):
        if self.dark_image is not None:
            return

        if self.darks is not None:
            dark_cache_filename = None
            min_files_to_cache = 4 # because our cache is float32 per sample, vs maybe 10 bits per sample for the source files
            if len(self.darks) >= min_files_to_cache and self.darks.start_raw_frame is 0 and self.darks.raw_frame_step is 1:
                dir = os.path.dirname(self.darks.filenames_and_raw_start_frames[0][0])
                # todo: could throw file modification times in here as well
                hash = hashlib.sha256(repr(self.darks.filenames_and_raw_start_frames).encode('utf-8')).hexdigest()
                basename = f'tensorez_dark_cache_{hash}.npy'
                dark_cache_filename = os.path.join(dir, basename)

            if dark_cache_filename is not None:
                try:
                    print(f"Loading dark image from {dark_cache_filename}")
                    self.dark_image = np.load(dark_cache_filename)
                    return
                except FileNotFoundError:
                    print("dark_image cache wasn't found.")

            print("Computing the average of the darks.")
            self.dark_image = self.darks.read_average_image()

            if dark_cache_filename is not None:
                print("Saving dark image cache to {dark_cache_filename}")
                np.save(dark_cache_filename, self.dark_image)

    def read_cooked_image(self, index):

        image = self.lights.read_image(index)

        self.load_or_create_dark_image()
        if self.dark_image is not None:
            image -= self.dark_image

        if self.crop_before_align and self.crop is not None:
            image = crop_image(image, crop=self.crop, crop_align=self.crop_align)

        if self.alignment_transforms is not None:
            image = align.transform_image(image, self.alignment_transforms[index])

        if self.align_by_center_of_mass:
            image = align_by_center_of_mass(image)

        if not self.crop_before_align and self.crop is not None:
            image = crop_image(image, crop=self.crop, crop_align=self.crop_align)

        return image

    def set_alignment_transforms(self, alignment_transforms):
        assert(alignment_transforms.shape[0] == len(self.lights))
        self.alignment_transforms = alignment_transforms

    def __getitem__(self, index):
        return self.read_cooked_image(index)

    def __len__(self):
        return len(self.lights)

    def __iter__(self):
        for i in range(len(self.lights)):
            yield self[i]
