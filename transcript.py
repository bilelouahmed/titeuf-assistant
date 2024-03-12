from utils import *
from talking import *
import speech_recognition as sr
import whisper
import torch
from datetime import datetime, timedelta
import numpy as np
import argparse
from queue import Queue
import os
from time import sleep
from sys import platform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="translation",
                        help="Mode considered by the assistant : \"translation\", \"chat\"", type=str)
    parser.add_argument("--model", default="base", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)

    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition."
                                 "Run this with 'list' to view available Microphones.", type=str)

    args = parser.parse_args()

    mode = args.mode

    if mode not in ["translation", "chat"]:
        raise ValueError("Mode argument does not comply (it must be equal to either 'translation' or 'chat').")
    
    if mode == "translation":
        source_language, target_language = choose_language("source"), choose_language("target")
        if source_language == "English":
            english = True
        else:
            english = False
    
    else:
        english = speak_english()

    phrase_time = None

    data_queue = Queue()

    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold

    recorder.dynamic_energy_threshold = False

    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    model = args.model
    if args.model != "large" and english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']
    answers = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")
    print("Start listening...")

    while True:
        try:
            now = datetime.utcnow()

            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True

                phrase_time = now
                
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)

                if mode == "translation":
                    answer = translation(text, source_language, target_language)
                    
                elif mode == "chat":
                    answer = generation(text)
                
                print(answer)
                talking(answer, target_language)
                answers.append(answer) 

                print('', end='', flush=True)

                sleep(0.5)
        except KeyboardInterrupt:
            break
    
    if mode == "translation":
        print("\nTranscription :")
        for line in transcription:
            print(line)

        print("\nTranslation :")
        for line in answers:
            print(line)
    
    else:
        print("\nTranscription :")
        for i in range(len(transcription)):
            print(transcription[i])
            print(answers[i])

if __name__ == "__main__":
    main()