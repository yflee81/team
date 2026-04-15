import sounddevice as sd
import whisper
from scipy.io.wavfile import write
import os

def recordaudio(target_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    fs = 16000
    duration = 5
    print('--- Recording Start ---')
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print('--- Recording End ---')

    write(target_path, fs, myrecording)
    
    # Confirming the file exists immediately after writing
    if os.path.exists(target_path):
        print(f"Successfully saved to: {target_path}")
    else:
        print("Failed to save the file. Check folder permissions.")

def SpeechToText(file_path: str) -> str:  
    if not os.path.exists(file_path):
        return f"Error: The file '{file_path}' does not exist on your disk."

    try:
        # fp16=False removes the CPU warning
        model = whisper.load_model("base")
        result = model.transcribe(file_path, fp16=False)
        return result["text"].strip()
    except Exception as e:
        return f"Error during transcription: {e}"

# Define the path once so there are no typos
save_path = r'C:\Users\UserAdmin\Downloads\Pythoncode\output.wav'

recordaudio(save_path)
x = SpeechToText(save_path)
print(f"Result: {x}")

