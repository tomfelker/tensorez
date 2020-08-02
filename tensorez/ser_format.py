import numpy as np
import mmap
import os
import struct
import collections
import enum

header_format = struct.Struct('<14slllllll40s40s40sqq')
if header_format.size != 178:
    raise "Hmm, size is wrong"

SerHeader = collections.namedtuple('SerHeader', 'file_id lu_id color_id little_endian image_width image_height pixel_depth_per_plane frame_count observer instrument telescope date_time date_time_utc')

class ColorId(enum.IntEnum):
    MONO = 0
    BAYER_RGGB = 8
    BAYER_GRBG = 9
    BAYER_GBRG = 10
    BAYER_BGGR = 11
    BAYER_CYYM = 16
    BAYER_YCMY = 17
    BAYER_YMCY = 18
    BAYER_MYYC = 19
    RGB = 100
    BGR = 101

def read_ser_header(filename):
    with open(filename, "rb") as file:
        header = SerHeader._make(header_format.unpack(file.read(header_format.size)))
        return header

def read_frame(filename, frame_index, to_float = True):
    with open(filename, "rb") as file:
        header = SerHeader._make(header_format.unpack(file.read(header_format.size)))

        num_channels = 1
        if header.color_id >= ColorId.RGB:
            num_channels = 3
        
        bytes_per_channel = header.pixel_depth_per_plane // 8
        bytes_per_pixel = bytes_per_channel * num_channels
        bytes_per_frame = bytes_per_pixel * header.image_width * header.image_height

        if bytes_per_channel == 1:
            dtype = np.dtype('uint8')
        elif bytes_per_channel == 2:
            dtype = np.dtype('uint16')
        else:
            raise 'too many bytes per channel'

        # hmm, this is opposite from the doc
        if header.little_endian == 0:
            dtype = dtype.newbyteorder('<')
        else:
            dtype = dtype.newbyteorder('>')

        file.seek(header_format.size + frame_index * bytes_per_frame)
        frame = file.read(bytes_per_frame)
    frame = np.frombuffer(frame, dtype = dtype)
    frame = frame.reshape((1, header.image_height, header.image_width, num_channels))

    if to_float:
        frame = frame.astype(np.float32)
        max_val = 1 << max(header.pixel_depth_per_plane, 8)
        frame = frame * (1 / max_val)

    return frame, header

if __name__ == '__main__':
    #read_frame(os.path.join("data", "ser_player_examples", "Jup_200415_204534_R_F0001-0300.ser"), 0)
    #read_frame(os.path.join("data", "ser_player_examples", "Mars_150414_002445_OSC_F0001-0500.ser"), 0)
    read_frame(os.path.join("data", 'ASICAP', 'CapObj', '2020-07-30Z', '2020-07-30-2020_6-CapObj.SER'), 0)
