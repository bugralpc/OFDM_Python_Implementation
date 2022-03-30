import OFDM_library
import numpy as np
import matplotlib.pyplot as plt

OFDM1 = OFDM_library.OFDMParameters(64, 16, 15000, 4, 4, 4)

print(OFDM1.OFDMFFTVector)

plt.plot(abs(OFDM1.OFDMFFTVector))
plt.show()

plt.plot(abs(OFDM1.CPOFDMVector))
plt.show()




