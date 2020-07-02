import numpy as np
import pydub
from pydub import AudioSegment


def wav_to_array_file(input, normalized=False):
    """Convert an at least 70 seconds long WAV-file to a numpy array"""
    a = pydub.AudioSegment.from_wav(
        input)  # get array from AudioSegment object
    slice = a[40000:70000]  # only take 30 sec. excerpt
    y = np.array(slice.get_array_of_samples())  # cast as NumPy array
    y_new = []

    if a.channels == 2:
        y = y.reshape((-1, 2))
        y_new = y[:, 0]

    else:
        y_new = y

    return y_new


def wav_to_array_rec(input):
    a = pydub.AudioSegment.from_wav(input)
    y = np.array(a.get_array_of_samples())

    return y


def mp3_to_array(f, normalized=False):
    """Convert an at least 70 seconds long MP3-file to a numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    slice = a[40000:70000]  # get a slice from 40 to 70 seconds of an mp3
    y = np.array(slice.get_array_of_samples())
    y_new = []
    if a.channels == 2:
        y = y.reshape((-1, 2))
        y_new = y[:, 0]

    else:
        y_new = y
    return y_new


def mp3_to_wav(input):
    """ Converts mp3 to WAV, renames 'filname.mp3' as 'filename.wav' """
    sound = AudioSegment.from_file(input, format='mp3')
    sound.export(name_wav(input), format='wav')
    print(name_wav(input))


def wav_to_mp3(input):
    """ Converts WAV to mp3, renames 'filname.wav' as 'filename.mp3' """
    sound = AudioSegment.from_file(input, format='wav')
    sound.export(name_mp3(input), format='mp3')


def name_wav(input):
    new_name = input[:-3] + "wav"  # change extension to 'wav'
    return new_name


def name_mp3(input):
    new_name = input[:-3] + "mp3"  # change extension to 'mp3'
    return new_name
