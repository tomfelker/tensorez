import glob
import fnmatch
import tensorez.ser_format as ser_format
from tensorez.util import read_image
import tensorflow as tf

class ImageSequence:
    # @param fileglobs is a list of, wildcard filenames of images or videos, or ImageSequences
    # @param start_frame is the first frame to use in the video
    # @param frame_step 
    # @param end_frame is the first frame that won't be included
    def __init__(self, fileglobs, start_frame = 0, frame_step = 1, end_frame = None):
        # raw refers to an index that could index any frame in any of the files,
        # cooked refers to frames relative to start_frame, and taking frame_step into account

        self.start_raw_frame = start_frame
        self.raw_frame_step = frame_step

        if isinstance(fileglobs, str) or isinstance(fileglobs, ImageSequence):
            fileglobs = [fileglobs]
        
        filenames = []
        for fileglob in fileglobs:
            for filename in glob.glob(fileglob):
                filenames.append(filename)

        self.filenames_and_raw_start_frames = []
        
        start_raw_frame_for_file = 0
        for filename in filenames:
            self.filenames_and_raw_start_frames.append((filename, start_raw_frame_for_file))
            start_raw_frame_for_file += self.get_frame_count(filename)

        self.raw_frame_count = start_raw_frame_for_file
        if end_frame is not None and end_frame < self.raw_frame_count:
            self.raw_frame_count = end_frame

        self.cooked_frame_count = int((self.raw_frame_count - 1 - self.start_raw_frame) / self.raw_frame_step) + 1

    def get_cache_hash_info(self):
        ret = "ImageSequence:\n"
        ret += "\tFiles:\n"
        for filename, start_raw_frame_for_file in self.filenames_and_raw_start_frames:
            ret += f"\t\t{filename}\n"
        ret += f"\tstart_raw_frame: {self.start_raw_frame}\n"
        ret += f"\traw_frame_step: {self.raw_frame_step}\n"
        ret += f"\traw_frame_count: {self.raw_frame_count}\n"
        return ret

    def cooked_to_raw_index(self, cooked_index):
        raw_index = self.start_raw_frame + cooked_index * self.raw_frame_step
        if raw_index < 0 or raw_index >= self.raw_frame_count:
            return None
        return raw_index

    def raw_to_cooked_index(self, raw_index):
        if raw_index < self.start_raw_frame or raw_index >= self.raw_frame_count:
            return None
        cooked_index = (raw_index - self.start_raw_frame) / self.raw_frame_step
        if cooked_index != int(cooked_index):
            return None
        return int(cooked_index)

    def raw_index_to_filename_and_frame_index(self, raw_index):
        filename = None
        frame_index = None
        for (possible_filename, start_raw_frame) in self.filenames_and_raw_start_frames:
            if start_raw_frame <= raw_index:
                filename = possible_filename
                frame_index = raw_index - start_raw_frame
            else:
                break
        return filename, frame_index

    def read_image(self, cooked_index, **kwargs):
        raw_index = self.cooked_to_raw_index(cooked_index)
        filename, frame_index_in_file = self.raw_index_to_filename_and_frame_index(raw_index)
        if isinstance(filename, ImageSequence):
            return filename.read_image(frame_index_in_file, **kwargs)
        return read_image(filename, frame_index = frame_index_in_file, **kwargs)

    @staticmethod
    def get_frame_count(filename):
        if fnmatch.fnmatch(filename, '*.ser'):
            ser_header = ser_format.read_ser_header(filename)
            return ser_header.frame_count
        # todo: other video types?
        if isinstance(filename, ImageSequence):
            return filename.cooked_frame_count
        # otherwise it's hopefully an image
        return 1

    class Iter:
        def __init__(self, image_sequence, read_image_kwargs = {}):
            self.image_sequence = image_sequence
            self.next_cooked_index = 0
            self.read_image_kwargs = read_image_kwargs

        def __iter__(self):
            return self

        def __next__(self):
            if self.image_sequence.cooked_to_raw_index(self.next_cooked_index) is None:
                raise StopIteration
            image = self.image_sequence.read_image(self.next_cooked_index, **self.read_image_kwargs)
            self.next_cooked_index += 1
            return image

    def __iter__(self):
        return ImageSequence.Iter(self)

    def __len__(self):
        return self.cooked_frame_count

    def __getitem__(self, index):
        return self.read_image(index)

    # this lets you do, e.g.:  for image in image_sequence.with_read_image_args(normalize = True)
    def with_read_image_args(self, **kwargs):
        return ImageSequence.Iter(self, kwargs)

    def read_average_image(self):
        average_image = None
        image_count = 0
        for image_hwc in self:
            if average_image is None:
                average_image = tf.Variable(tf.zeros_like(image_hwc))
                
            average_image.assign(average_image + image_hwc)

            image_count += 1
            print(f"Averaging, finished reading image {image_count} of {self.cooked_frame_count}")
            
        if image_count == 0:
            raise RuntimeError(f"Couldn't load any images.")
        average_image.assign(average_image * (1.0 / image_count))
        return average_image

