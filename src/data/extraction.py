#import essentia
#Code inspired by
# https://www.kaggle.com/code/papeloto/urban-sound-feature-extraction-knn?scriptVersionId=12264904&cellId=2

import os
import time

import numpy as np
import pandas as pd

import librosa
import librosa.display
import soundfile as sf # librosa fails when reading files on Kaggle.

import matplotlib.pyplot as plt
import IPython.display as ipd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
time1 = time.perf_counter()
audio_path = 'src\Audio\T\Becca_T.wav'
ipd.Audio(filename=audio_path,url = None, embed = None,rate = 44100, element_id=None )


x, sr = librosa.load(audio_path,sr=4)

plt.figure(figsize=(12, 5))
librosa.display.waveshow(x, sr=sr)
time2 = time.perf_counter()
#print(time2 - time1)
plt.show()

# plt.figure(figsize=(12, 5))
#plt.plot(x[1000:1100]) # Zoom-in for seeing the example.
# plt.grid()
# plt.show()

# n_crossings = librosa.zero_crossings(x[1000:1100], pad=False)
# print(f'Number of crosses: {sum(n_crossings)}')





