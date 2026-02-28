import speech_recognition as sr
import time
import keyboard
r = sr.Recognizer()
all_text = []

def callback(recognizer, audio):
    print(audio)
    text = recognizer.recognize_google(audio)
    with open("text.txt", 'w') as f:
        f.write(text)
    with open("audio.wav", 'wb') as w:
        w.write(audio.get_wav_data())

m = sr.Microphone()
with m as source:
    r.adjust_for_ambient_noise(source)
stop_listening = r.listen_in_background(m, callback)
print("Listening...")
keyboard.wait('q')
print("Stopping...")
stop_listening(wait_for_stop=True)