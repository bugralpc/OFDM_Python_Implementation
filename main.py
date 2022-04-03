import OFDM_library
import numpy as np
import matplotlib.pyplot as plt

OFDM1 = OFDM_library.OFDMParameters(64, 16, 15000, 4, 4, 4)

print(OFDM1.OFDMFFTVector)


plt.stem(abs(OFDM1.OFDMFFTVector))
plt.show()

plt.plot(abs(OFDM1.CPOFDMVector))
plt.show()

plt.scatter(np.real(OFDM1.OFDMFFTVector), np.imag(OFDM1.OFDMFFTVector))
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.show()

rxSymbols = OFDM_library.AddAWGNNoise(OFDM1.OFDMFFTVector, 0.005)

plt.scatter(np.real(rxSymbols), np.imag(rxSymbols))
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.show()

