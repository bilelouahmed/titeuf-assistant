import torch
from TTS.api import TTS
from utils import *
from pydub import AudioSegment
from pydub.playback import play
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

correspondance = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cz",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Hugarian": "hu",
    "Korean": "ko"
}

def talking(text:str, target_language:str) -> None:
    wav = tts.tts(text=text, language=correspondance[target_language])

    audio_bytes = b''.join([np.array(w).astype('float32').tobytes() for w in wav])

    audio = AudioSegment(
        audio_bytes,
        frame_rate=22050,
        sample_width=4,
        channels=1 
    )

    print("Start talking...")
    play(audio)