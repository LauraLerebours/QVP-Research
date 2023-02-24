import wave
import numpy as np
import matplotlib.pyplot as plt

wav_obj = wave.open('src\Audio\T\Becca_T.wav', 'rb')

sample_freq = wav_obj.getframerate()
samples = wav_obj.getnframes()
t_audio = samples/sample_freq
channels = wav_obj.getnchannels()
signal_wave = wav_obj.readframes(samples)
signal_array = np.frombuffer(signal_wave, dtype=np.float16)
l_channel = signal_array[0::2]
r_channel = signal_array[1::2]
times = np.linspace(0, samples/sample_freq, num=samples)
plt.figure(figsize=(15, 5))
plt.plot(times, l_channel)
plt.title('Left Channel')
plt.ylabel('Signal Value')
plt.xlabel('Time (s)')
plt.xlim(0, t_audio)
plt.show()