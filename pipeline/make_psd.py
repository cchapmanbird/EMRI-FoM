import numpy as np
from stableemrifisher.noise import noise_PSD_AE



freqs = np.linspace(0, 1, 100001)[1:]

psd = noise_PSD_AE(freqs, TDI = 'TDI2')

np.save("example_psd.npy",np.vstack((freqs, psd)).T)
