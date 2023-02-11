import tensorez.align as align
import tensorez.local_align as local_align
from tensorez.util import *
import tensorflow as tf
import hashlib
import os.path
import numpy as np
import gc

class Observation:
    def __init__(self, lights, darks = None, align_by_center_of_mass = False, align_by_content = False, local_align = False, crop = None, crop_align = 2, crop_before_align = False, crop_before_content_align = False, compute_alignment_transforms_kwargs = {}, align_by_center_of_mass_only_even_shifts = False):
        self.lights = lights
        self.darks = darks
        self.align_by_center_of_mass = align_by_center_of_mass
        self.align_by_content = align_by_content
        self.local_align = local_align
        self.compute_alignment_transforms_kwargs = compute_alignment_transforms_kwargs
        self.crop = crop
        self.crop_align = crop_align
        self.crop_before_align = crop_before_align
        self.crop_before_content_align = crop_before_content_align
        self.align_by_center_of_mass_only_even_shifts = align_by_center_of_mass_only_even_shifts

        self.dark_image = None
        self.alignment_transforms = None
        self.local_alignment_dataset = None

        self.debug_frame_limit = None

        self.computing_alignment_transforms = False

        self.cache_dir = 'cache'

        self.debug_output_dir = None

        # this should be all things that might affect the result, for use in logs and such so we can recreate things
        self.tensorez_steps = 'Observation with:\n'
        self.tensorez_steps += f'\tlights: {self.lights.get_cache_hash_info()}\n'
        if self.darks is not None:
            self.tensorez_steps += f'\tdarks: {self.darks.get_cache_hash_info()}\n'
        self.tensorez_steps += f'\talign_by_center_of_mass: {self.align_by_center_of_mass}\n'
        self.tensorez_steps += f'\talign_by_content: {self.align_by_content}\n'
        self.tensorez_steps += f'\tcompute_alignment_transforms_kwargs: {self.compute_alignment_transforms_kwargs}\n'
        self.tensorez_steps += f'\tcrop: {self.crop}\n'
        self.tensorez_steps += f'\tcrop_align: {self.crop_align}\n'
        self.tensorez_steps += f'\tcrop_before_align: {self.crop_before_align}\n'
        self.tensorez_steps += f'\tcrop_before_content_align: {self.crop_before_content_align}\n'
        self.tensorez_steps += f'\talign_by_center_of_mass_only_even_shifts: {align_by_center_of_mass_only_even_shifts}\n'

    def load_or_create_dark_image(self):
        if self.dark_image is not None:
            return

        if self.darks is not None:
            dark_cache_npy_filename = None
            min_files_to_cache = 4 # because our cache is float32 per sample, vs maybe 10 bits per sample for the source files
            if len(self.darks) >= min_files_to_cache:
                hash_info = self.darks.get_cache_hash_info()
                hash = hashlib.sha256(repr(hash_info).encode('utf-8')).hexdigest()

                dir = os.path.join(self.cache_dir, 'darks', hash)
                os.makedirs(dir, exist_ok=True)

                basename = 'dark_cache'
                basename_npy = basename + '.npz'
                basename_txt = basename + '.txt'
                dark_cache_npz_filename = os.path.join(dir, basename_npy)
                dark_cache_txt_filename = os.path.join(dir, basename_txt)

            if dark_cache_npz_filename is not None:
                try:                    
                    dark_dict = np.load(dark_cache_npz_filename)
                    self.dark_image = dark_dict['dark_image']
                    self.dark_variance = dark_dict['dark_variance']
                    print(f"Loaded dark image from '{dark_cache_npz_filename}'.")
                    return
                except FileNotFoundError:
                    print(f"Dark image cache wasn't found.  (Wanted '{dark_cache_npz_filename}')")

            print("Computing the average of the darks.")
            
            dark_variance_state = None
            for dark_frame_index, dark_frame in enumerate(self.darks):
                print(f'Averaging dark frame {dark_frame_index+1} of {len(self.darks)}')
                if dark_variance_state is None:
                    dark_variance_state = welfords_init(dark_frame.shape)
                dark_variance_state = welfords_update(dark_variance_state, dark_frame)                
            self.dark_image = welfords_get_mean(dark_variance_state)
            self.dark_variance = welfords_get_variance(dark_variance_state)
            del dark_variance_state
            gc.collect()
            # stdev can be inferred from that, if needed

            if dark_cache_npz_filename is not None:
                print(f"Saving dark image cache to {dark_cache_npz_filename}")
                np.savez_compressed(dark_cache_npz_filename, dark_image=self.dark_image, dark_variance=self.dark_variance)
                with open(dark_cache_txt_filename, mode='w') as txtfile:
                    txtfile.write(hash_info)

    def load_or_create_alignment_transforms(self):        
        if self.alignment_transforms is not None:
            return
        if len(self) <= 1:
            print("observation has no data, nothing to align.")
            return

        if self.align_by_content is False:
            return
        if self.computing_alignment_transforms is True:
            return

        # this should be only those things that  might affect the alignment transforms we compute (so not the same as tensorez_steps)
        lights_info = 'Lights:\n' + self.lights.get_cache_hash_info()
        hash_info = lights_info.replace('\n', '\n\t') + "\n"
        if self.darks is not None:
            darks_info = "Darks:\n" + self.darks.get_cache_hash_info()
            hash_info += darks_info.replace('\n', '\n\t') + "\n"
        if self.crop_before_align or self.crop_before_content_align:
            hash_info += f"Pre-align crop: {self.crop, self.crop_align}\n"
            hash_info += f"crop_before_align: {self.crop_before_align}\n"
            hash_info += f"crop_before_content_align: {self.crop_before_content_align}\n"
        hash_info += f"align_by_center_of_mass: {self.align_by_center_of_mass}\n"
        if self.align_by_center_of_mass:
            hash_info += f"align_by_center_of_mass_only_even_shifts: {self.align_by_center_of_mass_only_even_shifts}\n"
        hash_info += f"compute_alignment_transform({self.compute_alignment_transforms_kwargs})\n"
        hash_info += f"local_align:{self.local_align}\n"

        hash = hashlib.sha256(repr(hash_info).encode('utf-8')).hexdigest()
        
        dir = os.path.join(self.cache_dir, 'alignments', hash)
        os.makedirs(dir, exist_ok=True)

        basename = 'alignment_transforms'
        basename_npy = basename + '.npy'
        basename_txt = basename + '.txt'
        alignment_cache_npy_filename = os.path.join(dir, basename_npy)
        alignment_cache_txt_filename = os.path.join(dir, basename_txt)

        if self.local_align:
            flow_dataset_filename=os.path.join(dir, 'flow_dataset.npy')

        try:            
            self.alignment_transforms = np.load(alignment_cache_npy_filename)
            if self.local_align:
                self.local_alignment_dataset = np.lib.format.open_memmap(
                    filename=flow_dataset_filename,
                    mode='r',
                )

            print(f"Loaded alignment transforms from '{alignment_cache_npy_filename}'.")
            return
        except FileNotFoundError:
            print(f"Alignment transform cache wasn't found.  (Wanted '{alignment_cache_npy_filename}')")

        try:
            # we're passing ourselves in (so that the compute process can have the dark images subtracted, but we need to avoid infinite recursion
            self.computing_alignment_transforms = True            
            if self.local_align:
                self.alignment_transforms, self.local_alignment_dataset = local_align.local_align(lights = self, flow_dataset_filename=flow_dataset_filename, debug_output_dir=self.debug_output_dir, **self.compute_alignment_transforms_kwargs)                
                # the dataset dir is already written to at this point, but we'll save the alignment transforms below, and won't consider
                # things done until that point.  that gives a weak facsimile of atomicity
            else:
                self.alignment_transforms = align.compute_alignment_transforms(lights = self, debug_output_dir=self.debug_output_dir, **self.compute_alignment_transforms_kwargs)
        finally:
            self.computing_alignment_transforms = False


        print("Saving alignment transforms to {alignment_cache_npy_filename}")
        np.save(alignment_cache_npy_filename, self.alignment_transforms)
        with open(alignment_cache_txt_filename, mode='w') as txtfile:
            txtfile.write(hash_info)


    def read_cooked_image(self, index, skip_content_align = False, want_dark_variance = False):

        self.load_or_create_dark_image()
        self.load_or_create_alignment_transforms()

        image = self.lights.read_image(index)

        if want_dark_variance:
            dark_variance = self.dark_variance
        else:
            dark_variance = None
        
        if self.dark_image is not None:
            image -= self.dark_image

        if self.crop_before_align and self.crop is not None:
            image = crop_image(image, crop=self.crop, crop_align=self.crop_align)
            if dark_variance is not None:
                dark_variance = crop_image(dark_variance, crop=self.crop, crop_align=self.crop_align)

        if self.align_by_center_of_mass:
            if dark_variance is None:
                image, align_by_center_of_mass(image, only_even_shifts = self.align_by_center_of_mass_only_even_shifts)
            else:
                image, dark_variance = align_by_center_of_mass(image, only_even_shifts = self.align_by_center_of_mass_only_even_shifts, also_align_hwc=dark_variance)

        if self.crop_before_content_align and self.crop is not None:
            image = crop_image(image, crop=self.crop, crop_align=self.crop_align)
            if dark_variance is not None:
                dark_variance = crop_image(dark_variance, crop=self.crop, crop_align=self.crop_align)

        if self.alignment_transforms is not None and not skip_content_align:
            flow = None
            if self.local_alignment_dataset is not None:
                flow = self.local_alignment_dataset[index : index + 1, :, :, :]
                image = local_align.transform_image(image, self.alignment_transforms[index], flow_bihw = flow)
            else:
                image = align.transform_image(image, self.alignment_transforms[index])
            if dark_variance is not None:
                dark_variance = align.transform_image(dark_variance, self.alignment_transforms[index])

        # sigh, this is getting ugly...
        if not self.crop_before_align and not self.crop_before_content_align and self.crop is not None:
            assert not skip_content_align
            image = crop_image(image, crop=self.crop, crop_align=self.crop_align)
            if dark_variance is not None:
                dark_variance = crop_image(dark_variance, crop=self.crop, crop_align=self.crop_align)

        if want_dark_variance:
            return image, dark_variance
        return image

    def read_bayer_filter_unaligned(self):
        bayer_filter = self.lights.read_bayer_filter()
        if self.align_by_center_of_mass:
            assert self.align_by_center_of_mass_only_even_shifts, 'When doing Bayer processing, you must set align_by_center_of_mass_only_even_shifts, so that the bayer patterns of all images will line up.  You may also want to align_by_content in this case.'
        if self.crop is not None:
            # This may not be strictly necessary, but it keeps things simple... otherwise if we decompose
            # the image into bayer channels, different crops would have different bayer patterns.
            assert self.crop_align == 2
            bayer_filter = crop_image(bayer_filter, crop=self.crop, crop_align=self.crop_align)
        return bayer_filter

    def __getitem__(self, index):
        return self.read_cooked_image(index)

    def __len__(self):
        length = len(self.lights)
        if self.debug_frame_limit is not None:
            length = min(length, self.debug_frame_limit)
        return length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
