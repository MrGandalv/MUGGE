import numpy as np
import pydub
from pydub import AudioSegment


def wav_to_array_file(input_file, duration=30, offset=0, normalized=False):
    """Convert an at least 70 seconds long WAV-file to a numpy array"""
    a = pydub.AudioSegment.from_wav(
        input_file)  # get array from AudioSegment object
    # slice = a[40000:70000]  # only take 30 sec. excerpt
    slice = a[1000 * offset:1000 * (offset + duration)]
    y = np.array(slice.get_array_of_samples())  # cast as NumPy array

    if a.channels == 2:
        y = y.reshape((-1, 2))
        y_new = y[:, 0]

    else:
        y_new = y

    return np.array(y_new, dtype=float)


def wav_to_array_rec(input_file):
    a = pydub.AudioSegment.from_wav(input_file)
    y = np.array(a.get_array_of_samples())

    return y


def mp3_to_array(input_file, duration=30, offset=0, normalized=False):
    """Convert an at least 70 seconds long MP3-file to a numpy array"""
    a = pydub.AudioSegment.from_mp3(input_file)
    slice = a[1000 * offset:1000 * (offset + duration)]  # get a slice from 40 to 70 seconds of an mp3
    y = np.array(slice.get_array_of_samples())
    y_new = []
    if a.channels == 2:
        y = y.reshape((-1, 2))
        y_new = y[:, 0]

    else:
        y_new = y
    return np.array(y_new, dtype=float)


def mp3_to_wav(input_file):
    """ Converts mp3 to WAV, renames 'filname.mp3' as 'filename.wav' """
    sound = AudioSegment.from_file(input_file, format='mp3')
    sound.export(name_wav(input_file), format='wav')
    print(name_wav(input_file))


def wav_to_mp3(input_file):
    """ Converts WAV to mp3, renames 'filname.wav' as 'filename.mp3' """
    sound = AudioSegment.from_file(input_file, format='wav')
    sound.export(name_mp3(input_file), format='mp3')
    return name_mp3(input_file)

def name_wav(input_file):
    new_name = input_file[:-3] + "wav"  # change extension to 'wav'
    return new_name


def name_mp3(input_file):
    new_name = input_file[:-3] + "mp3"  # change extension to 'mp3'
    return new_name
