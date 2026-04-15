import sounddevice as sd
import whisper
from scipy.io.wavfile import write
import os

def recordaudio():
    fs = 16000
    duration = 5
    print('start')
    myrecording = sd.rec(int(duration * fs), samplerate = fs, channels = 1, dtype = 'int16')
    sd.wait()
    print('end')

    write('output.wav', fs, myrecording)

    
    

def SpeechToText(file_path: str) -> str:  
    try:

        model = whisper.load_model("medium")
        
        result = model.transcribe(file_path, fp16 = False)
        
        return result["text"].strip()
    except FileNotFoundError:
        return "Error: The specified file was not found."
    except Exception as e:
        return f"Error during transcription: {e}"

recordaudio()
file_path = 'output.wav'
x = SpeechToText(file_path)
print(x)

if os.path.exists(file_path):
    os.remove(file_path)
