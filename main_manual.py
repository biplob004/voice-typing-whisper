import sounddevice as sd
import numpy as np
from pynput import keyboard
from pynput.keyboard import Controller as KeyController
import threading
import time


import whisper, torch
from librosa import resample
import numpy as np

whisper_model = whisper.load_model("large-v3")

def speech_to_text(audio_data, sample_rate=16000):
    
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.squeeze().numpy()
    

    audio_data = audio_data.astype(np.float32)


    if sample_rate != 16000:
        audio_data = resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000


    audio_data = audio_data.astype(np.float32)
    result = whisper_model.transcribe(audio_data )
    return result["text"]




sample_rate = 16000 
channels = 1 
audio_data = []
is_recording = False


keyboard_controller = KeyController()


def start_recording():
    global is_recording, audio_data
    is_recording = True
    audio_data = []
    print("Recording started...")
    
    def callback(indata, frames, time, status):
        if is_recording:
            audio_data.append(indata.copy())
    
    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback):
        while is_recording:
            sd.sleep(100)


def stop_recording():
    global is_recording, audio_data
    is_recording = False
    print("Recording stopped.")
    
    if audio_data:
        audio_data = np.concatenate(audio_data, axis=0)
        
        audio_data = audio_data.reshape(-1)
        text = speech_to_text(audio_data)
        print(f"Converted text: {text}")
        # keyboard_controller.type(text)
        for char in text:
            keyboard_controller.type(char)
            time.sleep(0.01)


def on_press(key):
    if key == keyboard.Key.alt_r:
        threading.Thread(target=start_recording).start()

def on_release(key):
    if key == keyboard.Key.alt_r:
        stop_recording()
    if key == keyboard.Key.esc:
        print('Stopping KEY listener,')
        return False

print('started ...')
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()


