import numpy as np


def AddAWGNNoise(complexSymbols, noisePower):
    noise = (np.random.randn(len(complexSymbols))) + 1j*np.random.randn(len(complexSymbols))/np.sqrt(2)
    RxSymbols = complexSymbols + noise*np.sqrt(noisePower)
    return RxSymbols

def AddPhaseNoise(complexSymbols, phaseNoisePower):
    phaseNoise = np.random.randn(len(complexSymbols))*phaseNoisePower
    RxSymbols = complexSymbols*np.exp(1j*phaseNoise)
    return RxSymbols

class OFDMParameters:
    dataSCNumber = 0
    bitsPerSymbol = 0
    randomBits = []
    complexSymbols = []
    OFDMFFTVector = []
    OFDMIFFTVector = []
    CPOFDMVector = []

    def __init__(self, FFTSize, CPSize, subcarrierSpace, PSKOrder, leftGuardSCNumber, rightGuardSCNumber):
        self.FFTSize = FFTSize
        self.CPSize = CPSize
        self.subcarrierSpace = subcarrierSpace
        self.PSKOrder = PSKOrder
        self.leftGuardSCNumber = leftGuardSCNumber
        self.rightGuardSCNumber = rightGuardSCNumber

        self.dataSCNumber = self.FFTSize - (self.leftGuardSCNumber + self.rightGuardSCNumber)
        self.bitsPerSymbol = int(np.log2(PSKOrder))

        OFDMParameters.generateComplexSymbols(self.dataSCNumber, self.bitsPerSymbol, self.PSKOrder)
        self.generateOFDMSymbol(self.complexSymbols, self.FFTSize, self.CPSize, self.leftGuardSCNumber,
                                self.rightGuardSCNumber)

    @classmethod
    def generateComplexSymbols(cls, dataSCNumber, bitsPerSymbol, PSKOrder):
        cls.randomBits = np.random.binomial(n=1, p=0.5, size=(dataSCNumber * bitsPerSymbol))
        randomBitsGroup = cls.randomBits.reshape(int(len(cls.randomBits) / bitsPerSymbol), bitsPerSymbol)
        m = np.arange(0, PSKOrder)
        I = 1 / np.sqrt(2) * np.cos(m / PSKOrder * 2 * np.pi)
        Q = 1 / np.sqrt(2) * np.sin(m / PSKOrder * 2 * np.pi)
        constellationMap = I + 1j * Q

        if PSKOrder == 2:
            for i in range(0, len(randomBitsGroup)):
                if np.array_equal(randomBitsGroup[i], [0]):
                    cls.complexSymbols.append(constellationMap[0])
                else:
                    cls.complexSymbols.append(constellationMap[1])
        elif PSKOrder == 4:
            for i in range(0, len(randomBitsGroup)):
                if np.array_equal(randomBitsGroup[i], [0, 0]):
                    cls.complexSymbols.append(constellationMap[0])
                elif np.array_equal(randomBitsGroup[i], [0, 1]):
                    cls.complexSymbols.append(constellationMap[1])
                elif np.array_equal(randomBitsGroup[i], [1, 0]):
                    cls.complexSymbols.append(constellationMap[2])
                else:
                    cls.complexSymbols.append(constellationMap[3])
        elif PSKOrder == 8:
            for i in range(0, len(randomBitsGroup)):
                if np.array_equal(randomBitsGroup[i], [0, 0, 0]):
                    cls.complexSymbols.append(constellationMap[0])
                elif np.array_equal(randomBitsGroup[i], [0, 0, 1]):
                    cls.complexSymbols.append(constellationMap[1])
                elif np.array_equal(randomBitsGroup[i], [0, 1, 0]):
                    cls.complexSymbols.append(constellationMap[2])
                elif np.array_equal(randomBitsGroup[i], [0, 1, 1]):
                    cls.complexSymbols.append(constellationMap[3])
                elif np.array_equal(randomBitsGroup[i], [1, 0, 0]):
                    cls.complexSymbols.append(constellationMap[4])
                elif np.array_equal(randomBitsGroup[i], [1, 0, 1]):
                    cls.complexSymbols.append(constellationMap[5])
                elif np.array_equal(randomBitsGroup[i], [1, 1, 0]):
                    cls.complexSymbols.append(constellationMap[6])
                else:
                    cls.complexSymbols.append(constellationMap[7])
    @classmethod
    def generateOFDMSymbol(cls, complexSymbols, FFTSize, CPSize, leftGuardSCNumber, rightGuardSCNumber):
        allCarriers = np.arange(0, FFTSize)
        guardCarriers = np.hstack([np.arange(0, leftGuardSCNumber), np.arange((FFTSize - rightGuardSCNumber), FFTSize)])
        dataCarriers = np.delete(allCarriers, guardCarriers)

        cls.OFDMFFTVector = np.zeros(FFTSize, dtype=complex)
        cls.OFDMFFTVector[guardCarriers] = 0
        cls.OFDMFFTVector[dataCarriers] = complexSymbols

        cls.OFDMIFFTVector = np.fft.ifft(cls.OFDMFFTVector, FFTSize)

        CP = cls.OFDMIFFTVector[(FFTSize - CPSize): FFTSize]
        cls.CPOFDMVector = np.hstack([CP, cls.OFDMIFFTVector])
