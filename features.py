import numpy as np
import librosa
from config import SAMPLE_RATE, N_SAMPLES, N_MFCC, N_FFT, HOP

N_SEG = 5  # so khuc thoi gian dung cho dac trung ML

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

def ml_vector(y):
    # 1) MFCC + delta + delta2 cho tung khung (T x 39)
    seq = mfcc_sequence(y)
    # 2) Chi giu doan co tieng noi (bo khoang lang 2 dau)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    T = min(len(rms), len(seq))
    rms, seq = rms[:T], seq[:T]
    idx = np.where(rms > 0.1 * rms.max())[0] if rms.max() > 0 else np.arange(T)
    a, b = (idx[0], idx[-1] + 1) if len(idx) else (0, T)
    if b - a < N_SEG:
        a, b = 0, T
    speech = seq[a:b]
    # 3) Chia doan tieng noi thanh N_SEG khuc, lay trung binh moi khuc -> giu thu tu am
    parts = np.array_split(speech, N_SEG)
    seg_means = np.concatenate([p.mean(axis=0) for p in parts])
    # 4) Them do lech chuan toan doan va do dai tieng noi
    return np.concatenate([seg_means, speech.std(axis=0), [(b - a) / T]]).astype(np.float32)