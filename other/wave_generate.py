import numpy as np
from scipy.io.wavfile import read, write

sample_rate = 44100

# Waves

def sin(t, f, a=1):
    return np.sin(np.linspace(0, 2 * np.pi, int(sample_rate * t)) * f * t) * a

def machine(t, t_scale=0.1, f_range=[50, 2000], a=1):
    wav = np.array([])
    t_end = int(sample_rate * t)
    while wav.size <= t_end:
        t_ = np.random.random() * t_scale
        f_ = np.random.randint(*f_range)
        wav = np.append(wav, sin(t_, f_, a=a))
    return wav[:t_end]

def noise(t, a=1):
    return (np.random.random(int(sample_rate * t)) - 0.5) * a

def silence(t):
    return np.zeros([int(t * sample_rate)])

# Scales

def equal_temperament(n):
    return 2 ** ((n - 49) / 12) * 440

def c_maj(n, octave=4, tr=0):
    keys = [0, 2, 4, 5, 7, 9, 11]
    offset = 12 * (octave + n // len(keys)) + 1
    key = keys[n % len(keys)]
    return equal_temperament(key + offset + tr)

def c_min(n, octave=4, tr=0):
    keys = [0, 2, 3, 5, 7, 8, 11]
    offset = 12 * (octave + n // len(keys)) + 1
    key = keys[n % len(keys)]
    return equal_temperament(key + offset + tr)




sequences = []

for i in range(10):
    sequences.append(machine(0.5, t_scale=0.1))
    sequences.append(noise(0.1))

for i in range(50):
    t = np.random.choice([0.1, 0.2, 0.4], p=[0.4, 0.4, 0.2])
    n = np.random.randint(-10, 10)
    wav = sin(t, c_min(n + np.random.choice([4, 7], p=[0.4, 0.6]))) + sin(t, c_min(n))
    sequences.append(wav)

for i in range(10):
    sequences.append(noise(0.05, 0.2))
    sequences.append(noise(0.05, 0.8))

for tr in range(-3, 5):
    for i in range(8):
        sequences.append(sin(.05, c_min(i, tr=tr)) + sin(.05, c_min(i - 3, tr=tr)))

    for i in reversed(range(7)):
        sequences.append(sin(.05, c_min(i, tr=tr)) + sin(.05, c_min(i - 7, tr=tr)))

sequences.append(noise(1, 0.5))
for i in range(10):
    sequences.append(machine(0.2))
    sequences.append(silence(0.1))
    sequences.append(noise(0.4, 0.5))

sequence = np.concatenate(sequences)

write('test.wav', sample_rate, sequence)

