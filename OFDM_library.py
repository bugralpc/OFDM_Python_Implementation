import numpy as np


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

        if self.PSKOrder == 2:
            self.bitsPerSymbol = 1
        elif self.PSKOrder == 4:
            self.bitsPerSymbol = 2
        elif self.PSKOrder == 8:
            self.bitsPerSymbol = 3

        OFDMParameters.generateComplexSymbols(self.dataSCNumber, self.bitsPerSymbol)
        self.generateOFDMSymbol(self.complexSymbols, self.FFTSize, self.CPSize, self.leftGuardSCNumber,
                                self.rightGuardSCNumber)

    @classmethod
    def generateComplexSymbols(cls, dataSCNumber, bitsPerSymbol):
        cls.randomBits = np.random.binomial(n=1, p=0.5, size=(dataSCNumber * bitsPerSymbol))
        randomBitsGroup = cls.randomBits.reshape(int(len(cls.randomBits) / bitsPerSymbol), bitsPerSymbol)
        for i in range(0, len(randomBitsGroup)):
            if np.array_equal(randomBitsGroup[i], [0, 0]):
                cls.complexSymbols.append(+1 - 1j)
            elif np.array_equal(randomBitsGroup[i], [0, 1]):
                cls.complexSymbols.append(+1 + 1j)
            elif np.array_equal(randomBitsGroup[i], [1, 0]):
                cls.complexSymbols.append(-1 + 1j)
            else:
                cls.complexSymbols.append(-1 - 1j)

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
