import speech_recognition as sr
import time
import pyaudio
import wave
import keyboard

def transcribe(source):
    r = sr.Recognizer()
    with sr.AudioFile(source) as w:
        text = r.recognize_google(r.record(w))
        with open("text.txt", 'w') as f: 
            f.write(text)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
OUTPUT_FILE = "audio.wav"

audio = pyaudio.PyAudio()

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("Listening...")

frames = []

while True:
    data = stream.read(CHUNK)
    frames.append(data)

    if keyboard.is_pressed('q'):
        break

print("Stopping...")

stream.stop_stream()
stream.close()
audio.terminate()

wf = wave.open(OUTPUT_FILE, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

transcribe("audio.wav")

print("Stopped.")