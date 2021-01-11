import numpy
from numpy.fft import fft
from numpy import sin, cos, pi, ones, zeros, arange, r_, sqrt, mean

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def rFFT(x):
    """
    Recursive FFT implementation.

    References
      -- http://www.cse.uiuc.edu/iem/fft/rcrsvfft/
      -- "A Simple and Efficient FFT Implementation in C++"
          by Vlodymyr Myrnyy
    """
    
    n = len(x)

    if (n == 1):
	    return x

    w = getTwiddle(n)
    m = n/2;
    X = ones(m, float)*1j
    Y = ones(m, float)*1j
    
    for k in range(m):
        X[k] = x[2*k]
        Y[k] = x[2*k + 1] 

    X = rFFT(X)  
    Y = rFFT(Y) 

    F = ones(n, float)*1j
    for k in range(n):
        i = (k%m)
        F[k] = X[i] + w[k] * Y[i]

    return F

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def getTwiddle(NFFT=8):
    """Generate the twiddle factors"""

    W = r_[[1.0 + 1.0j]*NFFT]

    for k in range(NFFT):
        W[k] = cos(2.0*pi*k/NFFT) - 1.0j*sin(2.0*pi*k/NFFT)

    return W

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def DFT(x, N=8):
    """
    Use the direct definition of DFT for verification
    """
    y = [1.0 + 1.0j]*N
    y = r_[y]
    for n in range(N):
        wsum = 0 + 0j
        for k in range(N):
            wsum = wsum + (cos(2*pi*k*n/N) - (1.0j * sin(2*pi*k*n/N)))*x[k]
    
    y[n] = wsum
        
    return y

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_rfft(N      = 64,      # FFT order to test
              nStart = 0.2,     # Note aliased signal included
              nStep  = 2.1,     # Samples per period step
              pStep  = pi/4,    # Phase step size
              limErr = 10e-12,  # Error limit to check
              maxErr = 0        # Max difference
              ):
    """
    Use the built in numpy FFT functions and the direct
    implemenation of the DFT to verify the recursive FFT.

    This testbench verifies the different implementations are within
    a certain limit.  Because of the different implemenations the values
    could be slightly off (computer representation calculation error).
    """

    # Use test signal nStart:nStep:N samples per cycle
    for s in arange(nStart, N+nStep, nStep):
        for p in arange(0, pi+pStep, pStep):

            n = arange(N, 0, -1)
            x = cos(2*pi*n/s + p)

            xDFT = DFT(x,N)
            nFFT = fft(x,N)
            xFFT = rFFT(x)

            rmsErrD = sqrt(mean(abs(xDFT - xFFT))**2)
            rmsErrN = sqrt(mean(abs(nFFT - xFFT))**2)

            if rmsErrD > limErr or rmsErrN > limErr:
                print(s, p, "Error!", rmsErrD, rmsErrN)
                print(xDFT)
                print(nFFT)
                print(xFFT)

            if rmsErrD > maxErr:
                maxErr = rmsErrD
            elif rmsErrN > maxErr:
                maxErr = rmsErrN

    print("N %d maxErr = %f " % (N,maxErr))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If the module is run test a bunch of different size FFTs
if __name__ == '__main__':

    # The following is fairly exhaustive and will take some time
    # to run.
    tv = 2**arange(1,12)
    for nfft in tv:
        test_rfft(N=nfft)