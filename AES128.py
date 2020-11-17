import json
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Hash import keccak
from base64 import b64encode, b64decode
import signalAnalysis as dsp
import numpy as np
import scipy.io.wavfile as wav
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
#
#
#
#
# TIME FOR AES AND KECCAK!!!
#
#
#


result_shape = np.shape(mfcc_result)
print(result_shape)
mfcc_vector_length = result_shape[0]
mfcc_coefficients_number = result_shape[1]
print(mfcc_coefficients_number*mfcc_vector_length)

counter = 0
k = keccak.new(digest_bits=512)

for i in range(mfcc_vector_length):
    for j in range(mfcc_coefficients_number):
        counter += 1
        secret = str(mfcc_result[i][j])

        header = b"myheader"
        data = bytes(secret, 'utf-8')
        k.update(data)

        key = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_GCM)
        cipher.update(header)
        ciphertext, tag = cipher.encrypt_and_digest(data)

        json_k = ['nonce', 'header', 'ciphertext', 'tag']
        json_v = [b64encode(x).decode('utf-8') for x in (cipher.nonce, header, ciphertext, tag)]
        result = json.dumps(dict(zip(json_k, json_v)))
        print(result)

        # We assume that the key was securely shared beforehand
        try:
            b64 = json.loads(result)
            json_k = ['nonce', 'header', 'ciphertext', 'tag']
            jv = {k: b64decode(b64[k]) for k in json_k}

            cipher = AES.new(key, AES.MODE_GCM, nonce=jv['nonce'])
            cipher.update(jv['header'])
            plaintext = cipher.decrypt_and_verify(jv['ciphertext'], jv['tag'])
            print("The message was: " + str(plaintext))
        except ValueError as KeyError:
            print("Incorrect decryption")

endTime1 = time.perf_counter()

print("\nAlgorithm execution time is: " + str(round(endTime1-startTime, ndigits=5)) + " seconds.")
print("Coefficients hash (Keccak SHA-512): " + str(k.hexdigest()))
