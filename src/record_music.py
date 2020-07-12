import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

import os

# use the converter module
from converter_module import wav_to_array_file, wav_to_array_rec, mp3_to_array, mp3_to_wav, wav_to_mp3, name_wav, name_mp3



def prepro(input_file):
    format_type = input_file[-4:]  # find formatextension
    # dictionary instead of if-else to avoid restructuring in case of additional formats
    formats = {
        ".wav": wav_to_array_file,
        ".mp3": mp3_to_array
    }
    # Get the NumPy array of an audiofile by calling function from formats dictionary
    try:
        myarray = formats.get(format_type)(input_file)
        return myarray
    except TypeError:
        print("Invalid format")


def record(duration):
    """ Record as WAV then convert to NumPy array to return."""
    fs = 44100  # Sample rate
    seconds = duration  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    print(f"Recording for {seconds} sec.:")
    sd.wait()  # Wait until recording is finished
    print("Recording terminated")
    print(myrecording)
    write('output.wav', fs, myrecording)  # Save as WAV file
    my_new = wav_to_array_rec(os.getcwd()+'\\output.wav')  # get NumPy array"
    return my_new
