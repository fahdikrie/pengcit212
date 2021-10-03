"""
helper.py
Contains helper function for image processing in frequency domain.
Includes meshgrid, lowpass filter, highpass filter, notch filter, and paddedsize.

Inherited from Asdos Pengolahan Citra Ganjil 2019/2020
Edited by Asdos Pengolahan Citra Ganjil 2020/2021
"""

import numpy as np

def dftuv(M, N):
    """ DFTUV Computes meshgrid frequency matrices.
    [U, V] = DFTUV(M, N) computes meshgrid frequency matrices U and
    V. U and V are useful for computing frequency-domain filter 
    functions that can be used with DFTFILT. U and V are both M-by-N. """

    # Set up range of variables.
    u = np.arange(M)
    v = np.arange(N)

    # Compute the indices for use in meshgrid
    for i in range(M//2+1, M):
        u[i] = u[i] - M
    for i in range(N//2+1, N):
        v[i] = v[i] - N

    # Return the meshgrid arrays
    return np.meshgrid(v,u)

def lpfilter(tipe, M, N, D0, n = 1): 
    """ LPFILTER Computes frequency domain lowpass filters
    H = LPFILTER(TYPE, M, N, D0, n) creates the transfer function of
    a lowpass filter, H, of the specified TYPE and size (M-by-N).  To
    view the filter as an image or mesh plot, it should be centered
    using H = fftshift(H).
 
    Valid values for TYPE, D0, and n are:
 
    'ideal'    Ideal lowpass filter with cutoff frequency D0.  n need
               not be supplied.  D0 must be positive
 
    'btw'      Butterworth lowpass filter of order n, and cutoff D0.
               The default value for n is 1.0.  D0 must be positive.

    'gaussian' Gaussian lowpass filter with cutoff (standard deviation)
 	           D0.  n need not be supplied.  D0 must be positive. """

    # Use function dftuv to set up the meshgrid arrays needed for 
    # computing the required distances.
    U, V = dftuv(M, N)

    # Compute the distances D(U, V).
    D = np.sqrt(U**2+V**2)

    # Begin fiter computations.
    if (tipe == 'gaussian'):
        return np.exp(-(D**2)/(2*(D0**2)))
    elif (tipe == 'btw'):
        return 1/(1 + (D/D0)**(2*n))
    elif (tipe == 'ideal'):
        return (D <= D0)
    else:
        print("Invalid type.")

def hpfilter(tipe, M, N, D0, n = 1):
    """ HPFILTER Computes frequency domain highpass filters
    H = HPFILTER(TYPE, M, N, D0, n) creates the transfer function of
    a highpass filter, H, of the specified TYPE and size (M-by-N).
    Valid values for TYPE, D0, and n are:
 
    'ideal'     Ideal highpass filter with cutoff frequency D0.  n
                need not be supplied.  D0 must be positive
 
    'btw'       Butterworth highpass filter of order n, and cutoff D0.
                The default value for n is 1.0.  D0 must be positive.
 
    'gaussian'  Gaussian highpass filter with cutoff (standard deviation)
                D0.  n need not be supplied.  D0 must be positive. """
 

    # The transfer function Hhp of a highpass filter is 1 - Hlp,
    # where Hlp is the transfer function of the corresponding lowpass
    # filter.  Thus, we can use function lpfilter to generate highpass
    # filters.
	
    # Generate highpass filter.
    Hlp = lpfilter(tipe, M, N, D0, n)
    return 1 - Hlp

def notch(tipe, M, N, D0, x, y, n = 1):
    """ notch Computes frequency domain notch filters
    H = NOTCH(TYPE, M, N, D0, x, y, n) creates the transfer function of
    a notch filter, H, of the specified TYPE and size (M-by-N). centered at
    Column X, Row Y in an unshifted Fourier spectrum.
    Valid values for TYPE, D0, and n are:
 
    'ideal'     Ideal highpass filter with cutoff frequency D0.  n
                need not be supplied.  D0 must be positive
 
    'btw'       Butterworth highpass filter of order n, and cutoff D0.
                The default value for n is 1.0.  D0 must be positive.
 
    'gaussian'  Gaussian highpass filter with cutoff (standard deviation)
                D0.  n need not be supplied.  D0 must be positive. """

    # The transfer function Hhp of a highpass filter is 1 - Hlp,
    # where Hlp is the transfer function of the corresponding lowpass
    # filter.  Thus, we can use function lpfilter to generate highpass
    # filters.

    # Generate highpass filter.
    Hlp = lpfilter(tipe, M, N, D0, n)
    H = 1 - Hlp
    H = np.roll(H, y-1, axis=0)
    H = np.roll(H, x-1, axis=1)
    return H

def paddedsize(w, l):
    """ PADDEDSIZE Computes padded sizes useful for FFT-based filtering.
    PQ = PADDEDSIZE(AB), where AB is a two-element size vector,
    computes the two-element size vector PQ = 2*AB. """

    return (2*w, 2*l)