import numpy as np
import librosa
from config import SAMPLE_RATE, N_SAMPLES, N_MFCC, N_FFT, HOP

def fix_length(y):
    if len(y) > N_SAMPLES:
        y = y[:N_SAMPLES]
    elif len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)

def load_wav(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return fix_length(y)

def mfcc_sequence(y):
    m = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    d1 = librosa.feature.delta(m)
    d2 = librosa.feature.delta(m, order=2)
    return np.concatenate([m, d1, d2], axis=0).T.astype(np.float32)

def mfcc_vector(seq):
    return np.concatenate([seq.mean(axis=0), seq.std(axis=0)]).astype(np.float32)