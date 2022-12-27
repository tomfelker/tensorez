import tensorez.align as align
from tensorez.util import *
import tensorflow as tf
import hashlib
import os.path
import numpy as np

class Observation:
    def __init__(self, lights, darks = None, align_by_center_of_mass = False, align_by_content = False, crop = None, crop_align = 1, crop_before_align = False, compute_alignment_transforms_kwargs = {}):
        self.lights = lights
        self.darks = darks
        self.align_by_center_of_mass = align_by_center_of_mass
        self.align_by_content = align_by_content
        self.compute_alignment_transforms_kwargs = compute_alignment_transforms_kwargs
        self.crop = crop
        self.crop_align = crop_align
        self.crop_before_align = crop_before_align

        self.dark_image = None
        self.alignment_transforms = None

        self.computing_alignment_transforms = False

    def load_or_create_dark_image(self):
        if self.dark_image is not None:
            return

        if self.darks is not None:
            dark_cache_npy_filename = None
            min_files_to_cache = 4 # because our cache is float32 per sample, vs maybe 10 bits per sample for the source files
            if len(self.darks) >= min_files_to_cache:
                dir = os.path.dirname(self.darks.filenames_and_raw_start_frames[0][0])
                # todo: could throw file modification times in here as well
                hash_info = self.darks.get_cache_hash_info()
                hash = hashlib.sha256(repr(hash_info).encode('utf-8')).hexdigest()
                basename = 'tensorez_dark_cache_' + hash
                basename_npy = basename + '.npy'
                basename_txt = basename + '.txt'
                dark_cache_npy_filename = os.path.join(dir, basename_npy)
                dark_cache_txt_filename = os.path.join(dir, basename_txt)

            if dark_cache_npy_filename is not None:
                try:                    
                    self.dark_image = np.load(dark_cache_npy_filename)
                    print(f"Loaded dark image from '{dark_cache_npy_filename}'.")
                    return
                except FileNotFoundError:
                    print(f"Dark image cache wasn't found.  (Wanted '{dark_cache_npy_filename}')")

            print("Computing the average of the darks.")
            self.dark_image = self.darks.read_average_image()

            if dark_cache_npy_filename is not None:
                print(f"Saving dark image cache to {dark_cache_npy_filename}")
                np.save(dark_cache_npy_filename, self.dark_image)
                with open(dark_cache_txt_filename, mode='w') as txtfile:
                    txtfile.write(hash_info)

    def load_or_create_alignment_transforms(self):
        if self.alignment_transforms is not None:
            return
        if len(self) <= 1:
            return
        if self.align_by_content is False:
            return
        if self.computing_alignment_transforms is True:
            return

        lights_info = 'Lights:\n' + self.lights.get_cache_hash_info()
        hash_info = lights_info.replace('\n', '\n\t') + "\n"
        if self.darks is not None:
            darks_info = "Darks:\n" + self.darks.get_cache_hash_info()
            hash_info += darks_info.replace('\n', '\n\t') + "\n"
        if self.crop_before_align:
            hash_info += f"Pre-align crop: {self.crop, self.crop_align}\n"
        hash_info += f"align_by_center_of_mass: {self.align_by_center_of_mass}\n"
        hash_info += f"compute_alignment_transform({self.compute_alignment_transforms_kwargs})\n"

        hash = hashlib.sha256(repr(hash_info).encode('utf-8')).hexdigest()
        dir = os.path.dirname(self.lights.filenames_and_raw_start_frames[0][0])
        basename = 'tensorez_alignment_cache_' + hash
        basename_npy = basename + '.npy'
        basename_txt = basename + '.txt'
        alignment_cache_npy_filename = os.path.join(dir, basename_npy)
        alignment_cache_txt_filename = os.path.join(dir, basename_txt)

        try:            
            self.alignment_transforms = np.load(alignment_cache_npy_filename)
            print(f"Loaded alignment transforms from '{alignment_cache_npy_filename}'.")
            return
        except FileNotFoundError:
            print(f"Alignment transform cache wasn't found.  (Wanted '{alignment_cache_npy_filename}')")

        try:
            # we're passing ourselves in (so that the compute process can have the dark images subtracted, but we need to avoid infinite recursion
            self.computing_alignment_transforms = True            
            self.alignment_transforms = align.compute_alignment_transforms(lights = self, **self.compute_alignment_transforms_kwargs)
        finally:
            self.computing_alignment_transforms = False


        print("Saving alignment transforms to {alignment_cache_npy_filename}")
        np.save(alignment_cache_npy_filename, self.alignment_transforms)
        with open(alignment_cache_txt_filename, mode='w') as txtfile:
            txtfile.write(hash_info)


    def read_cooked_image(self, index):

        self.load_or_create_dark_image()
        self.load_or_create_alignment_transforms()

        image = self.lights.read_image(index)
        
        if self.dark_image is not None:
            image -= self.dark_image

        if self.crop_before_align and self.crop is not None:
            image = crop_image(image, crop=self.crop, crop_align=self.crop_align)

        if self.align_by_center_of_mass:
            image = align_by_center_of_mass(image)

        if self.alignment_transforms is not None:
            image = align.transform_image(image, self.alignment_transforms[index])

        if not self.crop_before_align and self.crop is not None:
            image = crop_image(image, crop=self.crop, crop_align=self.crop_align)

        return image

    def __getitem__(self, index):
        return self.read_cooked_image(index)

    def __len__(self):
        return len(self.lights)

    def __iter__(self):
        for i in range(len(self.lights)):
            yield self[i]
