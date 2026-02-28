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

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("Could not understand audio")

    except KeyboardInterrupt:
        print("Program terminated by user")
        break

