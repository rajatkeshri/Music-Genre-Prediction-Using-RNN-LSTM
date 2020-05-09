from os import path
from pydub import AudioSegment



def convert_to_wav(src,dst):
    # convert wav to mp3
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")
