from pydub import AudioSegment
from pydub.playback import play
import pydub
import sounddevice as sd
import soundfile as sf


def play_wav(input):
    sound = AudioSegment.from_wav(input)
    play(sound)


def play_mp3(input):
    sound = AudioSegment.from_mp3(input)
    play(sound)
