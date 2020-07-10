import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# use the converter module
import converter as conv


def prepro(input_file):
    format_type = input_file[-4:]  # find formatextension
    # dictionary instead of if-else to avoid restructuring in case of additional formats
    formats = {
        ".wav": conv.wav_to_array_file,
        ".mp3": conv.mp3_to_array
    }
    # Get the NumPy array of an audiofile by calling function from formats dictionary
    try:
        myarray = formats.get(format_type)(input_file)
        return myarray
    except TypeError:
        print("Invalid format")


def record():
    """ Record as WAV then convert to NumPy array to return."""
    fs = 44100  # Sample rate
    seconds = 30  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    print("Recording  for 30 sec.:")
    sd.wait()  # Wait until recording is finished
    print("Recording terminated")
    write('output.wav', fs, myrecording)  # Save as WAV file
    my_new = conv.wav_to_array_rec('output.wav')  # get NumPy array"

    return my_new
