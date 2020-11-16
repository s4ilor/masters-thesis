import signalAnalysis as dsp
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import time


startTime = time.perf_counter()

# SETTING PATH TO THE FILE
filepath = "/Users/sailor/masters_thesis/data/pozoga_julita.wav"

# SETTING SIGNAL DSP DATA (timeframe: length (ms) of single frame, alpha: pre-emphasis coefficient)
timeframe = 0.02
alpha = 0.96

# EXTRACTING THE SAMPLING RATE AND AUDIO SIGNAL
(rate, signal) = wav.read(filepath)

# SETTING UPPER FREQUENCY FOR FILTERING
upperfreq = 5000

# REMOVING THE SILENCE AT BEGINNING AND ENDING OF THE SIGNAL
entireSignal = signal
signal = dsp.removesilence(signal)
last_sample = len(signal)
last_sample_entireSignal = len(entireSignal)

# FILTERING THE SIGNAL (BUTTERWORTH 2nd ORDER FILTER)
filtered_signal = dsp.frequency_filter(signal, rate, highcutfreq=upperfreq)

# RAW SIGNAL PRE-EMPHASIS
emphasized_signal = dsp.preemphasis(signal, alpha=alpha)

# FILTERED SIGNAL PRE-EMPHASIS
emphasized_filtered_signal = dsp.preemphasis(filtered_signal, alpha=alpha)

# FRAMING AND WINDOWING THE CLEANED SIGNAL (FILTERED AND PRE-EMPHASIZED)
framed_signal, windowed_signal = dsp.hamming_window(emphasized_filtered_signal, rate=rate, timeframe=timeframe)

# FLATTENING SIGNAL
windowed_signal_flat = np.ndarray.flatten(windowed_signal)

# SPECTRUM DATA EXTRACTION
xf, abs_fft, signal_flat_fft, Nx, xfx, freq, signal_avg_dft, f, Pxx_den, N, psd = dsp.spectrum(windowed_signal, rate=rate)

mfcc_result = dsp.mel_frequency_cepstral_coefficients(windowed_signal_flat, rate=rate)

endTime1 = time.perf_counter()


plt.figure()
plt.title("Raw signal")
plt.xlabel("Sample $s[n]$")
plt.ylabel("Amplitude")
plt.grid()
plt.ylim([-15000, 15000])
plt.xlim([0, last_sample])
plt.plot(signal)
plt.show()

plt.figure()
plt.title("Comparison - raw and filtered signals")
plt.xlabel("Sample $s[n]$")
plt.ylabel("Amplitude")
plt.grid()
plt.ylim([-15000, 15000])
plt.xlim([0, last_sample])
plt.plot(signal)
plt.plot(filtered_signal)
plt.show()

plt.figure()
plt.title("Raw signal after pre-emphasis")
plt.xlabel("Sample $s[n]$")
plt.ylabel("Amplitude")
plt.grid()
plt.ylim([-2000, 2000])
plt.xlim([0, last_sample])
plt.plot(emphasized_signal)
plt.show()

plt.figure()
plt.title("Comparison - raw and filtered signals after pre-emphasis")
plt.xlabel("Sample $s[n]$")
plt.ylabel("Amplitude")
plt.grid()
plt.ylim([-2000, 2000])
plt.xlim([0, last_sample])
plt.plot(emphasized_signal)
plt.plot(emphasized_filtered_signal)
plt.show()

plt.figure()
plt.title("Comparison - filtered signal before and after pre-emphasis")
plt.xlabel("Sample $s[n]$")
plt.ylabel("Amplitude")
plt.grid()
plt.ylim([-8000, 9000])
plt.xlim([0, last_sample])
plt.plot(filtered_signal)
plt.plot(emphasized_filtered_signal)
plt.show()

plt.figure()
plt.plot(framed_signal)
plt.ylim([-2000, 2000])
plt.xlim([0, len(framed_signal)])
plt.title('Reconstructed signal from frames')
plt.xlabel('Frame $f[k]$')
plt.ylabel('Amplitude')
plt.show()

plt.figure()
plt.plot(windowed_signal_flat)
plt.ylim([-1500, 1500])
plt.xlim([0, len(windowed_signal_flat)])
plt.title('Reconstructed signal from windowed frames (flattened)')
plt.grid()
plt.xlabel('Window $w[k]$')
plt.ylabel('Amplitude')
plt.show()

plt.figure()
plt.plot(framed_signal[4])
plt.xlim([0, len(framed_signal[4])])
plt.ylim([-1500, 1500])
plt.title('Zoom on 4th frame before windowing')
plt.grid()
plt.xlabel('Frame samples $s(f=150)[n]$')
plt.ylabel('Amplitude')
plt.show()

plt.figure()
plt.plot(windowed_signal[4])
plt.xlim([0, len(windowed_signal[4])])
plt.ylim([-1500, 1500])
plt.title('Zoom on 4th frame after windowing')
plt.grid()
plt.xlabel('Window samples $s(f=150)[n]$')
plt.ylabel('Amplitude')
plt.show()

plt.figure()
plt.grid()
plt.xlim([0, 4500])
plt.ylim([0, 600000])
plt.plot(xfx, signal_flat_fft[:Nx // 2])
plt.title('Zoomed DFT in range 0 - 4.5kHz')
plt.xlabel('Frequency in [Hz]')
plt.ylabel('Amplitude')
plt.show()

plt.figure()
plt.grid()
plt.xlim([0, 2000])
plt.ylim([0, 600000])
plt.plot(xfx, signal_flat_fft[:Nx // 2])
plt.title('Zoomed DFT in range 0 - 2kHz')
plt.xlabel('Frequency in [Hz]')
plt.ylabel('Amplitude')
plt.show()

plt.figure()
plt.title('Average DFT (Bartlett method)')
plt.xlabel('Frequency in [Hz]')
plt.ylabel('Amplitude')
plt.plot(freq, np.abs(signal_avg_dft))
plt.xlim([0, 2000])
plt.ylim([0, 32000])
plt.grid()
plt.show()

plt.figure()
plt.grid()
plt.xlim([0, 4500])
plt.plot(xf, np.power(abs_fft[4][:N//2]/960, 2))
plt.title('$ S_i(k) = |DFT|^2 * 1/N$ for 4th window')
plt.xlabel('Frequency in [Hz]')
plt.ylabel('Amplitude')
plt.show()

plt.figure()
plt.title('Average Periodogram (Bartlett method)')
plt.plot(freq, np.abs(psd))
plt.xlabel('Freq (Hz)')
plt.ylabel('Amplitude')
plt.xlim([0, 4500])
plt.grid()
plt.show()

plt.semilogy(f, np.abs(Pxx_den))
plt.grid()
plt.title('Power Spectral Density')
plt.xlabel('Frequency $[Hz]$')
plt.ylabel('PSD $V^2/Hz$')
plt.xlim([0, 10000])
plt.show()

plt.figure()
plt.title('MFCC coefficients for entire signal')
plt.xlabel('Frame number')
plt.ylabel('Coefficient number')
plt.imshow(mfcc_result.T, cmap='rainbow')
plt.show()

endTime2 = time.perf_counter()

print("\nAlgorithm execution time (without images) is: " + str(round(endTime1-startTime, ndigits=5)) + " seconds.")
print("Algorithm execution time (with images) is: " + str(round(endTime2-startTime, ndigits=5)) + " seconds.")