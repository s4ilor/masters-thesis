import math
import numpy as np
import scipy.fftpack
from python_speech_features import mfcc
from scipy import signal


def frequency_filter(inputsignal, rate, highcutfreq):
    sos = signal.butter(N=2, Wn=highcutfreq, btype='low', fs=rate, analog=False, output='sos')
    filtered_signal = signal.sosfilt(sos, inputsignal)
    return filtered_signal


def preemphasis(signal, alpha=0.96):
    emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    return emphasized_signal


def hamming_window(emphasized_filtered_signal, rate=48000, timeframe=0.02):
    signal = emphasized_filtered_signal
    length = len(emphasized_filtered_signal)

    # 20ms is a default timeframe for a voice signal
    # 20ms * 48kHz = 960 samples per 1 frame
    frame_length = int(rate * timeframe)

    # Framing process
    if length % frame_length == 0:
        frames_amount = int(length / frame_length)
    else:
        frames_amount = int(math.floor(length / frame_length))
        finish = int(frames_amount * frame_length)
        signal = signal[0:finish]
    framed_signal = np.zeros(shape=(frames_amount, frame_length))
    for k in range(frames_amount):
        framed_signal[k, :] = signal[0 + k * frame_length: frame_length + k * frame_length]

    # Windowing process
    windowed_signal = np.zeros(shape=(frames_amount, frame_length))
    for i in range(0, frames_amount):
        windowed_signal[i, :] = framed_signal[i] * np.hamming(frame_length)

    return framed_signal, windowed_signal


def spectrum(windowed_signal, rate=48000):
    # Importing signal from main_window
    signal = windowed_signal

    # Basics
    T = 1 / rate
    N = len(signal[0])
    xf = np.linspace(0.0, 1.0 / (2.0 * T), int(math.floor(N / 2)))

    # DFT calculations
    window_fft = scipy.fftpack.fft(signal)
    abs_fft = np.abs(window_fft)

    signal_flat = np.ndarray.flatten(np.asarray(signal))

    Nx = len(signal_flat)
    xfx = np.linspace(0.0, 1.0 / (2.0 * T), int(math.floor(Nx // 2)))

    # DFT for the "new" flattened signal
    signal_flat_fft = np.abs(scipy.fftpack.fft(signal_flat))

    # Bartlett method - method of averaged periodograms
    n_fft = 960
    periodogram = np.zeros(n_fft, dtype='complex128')
    spectrogram = np.zeros(n_fft, dtype='complex128')
    total_segments = len(signal_flat) // n_fft

    for i in range(total_segments):
        p1 = i * n_fft  # segment start position
        p2 = p1 + n_fft  # segment end position
        segment = signal_flat[p1: p2]
        periodogram += np.abs(np.fft.fft(segment)) ** 2 / n_fft
        spectrogram += np.abs(np.fft.fft(segment))

    psd = periodogram / total_segments  # average periodogram
    psd = psd[0: n_fft // 2]
    freq = np.linspace(0, rate, n_fft)  # calc frequency axis
    freq = freq[0: n_fft // 2]

    signal_avg_dft = spectrogram / total_segments
    signal_avg_dft = signal_avg_dft[0: n_fft // 2]

    f, Pxx_den = scipy.signal.periodogram(signal_flat, fs=48000, window='flattop', scaling='spectrum')

    return xf, abs_fft, signal_flat_fft, Nx, xfx, freq, signal_avg_dft, f, Pxx_den, N, psd


def mel_frequency_cepstral_coefficients(windowed_signal_flat, rate=48000):
    signal = windowed_signal_flat
    mfcc_result = mfcc(signal, samplerate=rate,
                       winlen=0.02, winstep=0.01,
                       nfilt=24, nfft=960,
                       lowfreq=0, highfreq=4000, preemph=0.96)
    return mfcc_result
