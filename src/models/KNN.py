#Code inspired by
# https://www.kaggle.com/code/papeloto/urban-sound-feature-extraction-knn?scriptVersionId=12264904&cellId=2

import os

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

audio_path = '\lereb\Downloads\Audio\T\Becca T.wav'
ipd.Audio(audio_path,44100)

x, sr = librosa.load(audio_path,sr=4)

plt.figure(figsize=(12, 5))
librosa.display.waveplot(x, sr=sr)
plt.show()
