import sounddevice as sd

fs = 16000
duration = 5
print('start')
myrecording = sd.rec((duration * fs), samplerate = fs, channels = 2)
sd.wait()
print('end')

sd.play(myrecording, fs)
sd.wait()
print('done')