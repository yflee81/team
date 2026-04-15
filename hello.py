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
    print(f"saved as {os.path.abspath('output.wav')}")

    
    

def SpeechToText(file_path: str) -> str:  
    try:
        # Load the model (options: 'tiny', 'base', 'small', 'medium', 'large')
        # The model is downloaded the first time you run this.
        model = whisper.load_model("base")
        
        # Transcribe the audio file
        result = model.transcribe(file_path)
        
        return result["text"].strip()
    except FileNotFoundError:
        return "Error: The specified file was not found."
    except Exception as e:
        return f"Error during transcription: {e}"

recordaudio()
file_path = 'output.wav'
x = SpeechToText(file_path)
print(x)

# Delete the file after processing is complete
if os.path.exists(file_path):
    os.remove(file_path)
