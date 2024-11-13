import numpy as np
import scipy.signal as sig
from scipy.fft import fft, fftfreq

def linelength(filtered_window):
    '''
    Input: 
     filtered_window: ECoG window after filtering, (window_samples x channels)

    Output: 
     Line length feature of the window (channels,)
    '''
    return np.sum(np.abs(np.diff(filtered_window, axis=0)), axis=0)

# energy
def energy(filtered_window):
    return np.sum(filtered_window ** 2, axis = 0)

def ZeroCrossing(filtered_window):
    return np.sum((np.logical_xor((filtered_window[:-1] - np.mean(filtered_window)) > 0, (filtered_window[1:] - np.mean(filtered_window)) > 0)), axis = 0)

### Katz Fractal Dimension hasn't been applied to Final Project Part1
def katz_fractal_dimension(filtered_window):
    """
    Calculate the Katz Fractal Dimension of a dataset.

    Parameters:
    data (numpy.ndarray): 1D array representing the data points.

    Returns:
    float: The Katz Fractal Dimension.
    """
    N = len(filtered_window)
    L = np.sum(np.abs(np.diff(filtered_window)))  # Calculate the total length of the section
    d = np.abs(filtered_window[0] - filtered_window[-1])  # Calculate the diameter of the section

    fractal_dimension = np.log(N - 1) / (np.log(d / L) + np.log(N - 1))

    return fractal_dimension

def time_average(filtered_window):
    '''
    Input: 
     filtered_window: ECoG window after filtering, (window_samples x channels)

    Output: 
     Average time-domain volatge of the window (channels,)
    '''
    return np.mean(filtered_window,axis=0)

def time_var(filtered_window):
    '''
    Input: 
     filtered_window: ECoG window after filtering, (window_samples x channels)

    Output: 
     Variance of time-domain volatge of the window (channels,)
    '''
    return np.var(filtered_window,axis=0)

def area(filtered_window):
    '''
    Input: 
     filtered_window: ECoG window after filtering, (window_samples x channels)

    Output: 
     Average time-domain volatge of the window (channels,)
    '''
    return np.sum(np.abs(filtered_window),axis=0)

def total_power(filtered_window,fs):
    '''
    Input: 
     filtered_window: ECoG window after filtering, (window_samples x channels)
     fs: sampling rate of ECoG (Hz)

    Output: 
     Total spectral power of the window (channels,)
    '''
    freq, psd = sig.welch(filtered_window, fs, nperseg=filtered_window.shape[0], axis=0)
    total_power = np.trapz(psd, freq, axis=0)

    return total_power

def bandpower(filtered_window,fs,fmin,fmax):
    freq, psd = sig.welch(filtered_window, fs, nperseg=filtered_window.shape[0], axis=0)
    idx_fmin = (np.abs(freq-fmin)).argmin()
    idx_fmax = (np.abs(freq-fmax)).argmin()
    band_oi_power = np.trapz(psd[idx_fmin:idx_fmax],freq[idx_fmin:idx_fmax],axis=0)
    return band_oi_power

def rel_bandpower(filtered_window,fs,fmin,fmax):
    '''
    Input: 
     filtered_window: ECoG window after filtering, (window_samples x channels)
     fs: sampling rate of ECoG (Hz)
     fmin: the lower bound of the band (Hz)
     fmax: the upper bound of the band (Hz)

    Output: 
     Relative bandpower of a frequency band of the window (channels,), and the total power of the window (channels,)
    '''
    freq, psd = sig.welch(filtered_window, fs, nperseg=filtered_window.shape[0], axis=0)
    total_power = np.trapz(psd, freq, axis=0)
    idx_fmin = (np.abs(freq-fmin)).argmin()
    idx_fmax = (np.abs(freq-fmax)).argmin()
    band_oi_power = np.trapz(psd[idx_fmin:idx_fmax],freq[idx_fmin:idx_fmax],axis=0)
    rel_band_power = band_oi_power/total_power
    return rel_band_power
    # return band_oi_power/total_power

def filt_amp(filtered_window,fs,fmin,fmax):
    '''
    Input: 
     filtered_window: ECoG window after filtering, (window_samples x channels)
     fs: sampling rate of ECoG (Hz)
     fmin: the lower bound of the band (Hz)
     fmax: the upper bound of the band (Hz)

    Output: 
     Average sqaure voltage of a particular frequency range of the window (channels,)
    '''
    b,a = sig.butter(4,[2*fmin/fs,2*fmax/fs],btype='band')
    x_filt = sig.filtfilt(b,a,filtered_window,axis=0,padlen=10)
    return np.mean(x_filt**2,axis=0)

def freq_average(filtered_window,fs,fmin,fmax):
    '''N = filtered_window.shape[0]
    fft_data = fft(filtered_window,axis=0)
    xf = fftfreq(N, 1/fs)[:N//2]
    idx = (int(2*fmin/fs*xf.shape[0]),int(2*fmax/fs*xf.shape[0]))
    return np.mean(np.abs(fft_data[idx[0]:idx[1]]),axis=0)'''
    
    N = filtered_window.shape[0]
    fft_data = fft(filtered_window, axis=0)
    xf = fftfreq(N, 1/fs)[:N//2]
    idx = np.where((xf >= fmin) & (xf <= fmax))[0]  # Find indices within the frequency band
    return np.mean(np.abs(fft_data[idx]), axis=0)

'''def avg_peak_to_peak(filtered_window):
    return np.mean(np.max(filtered_window, axis=0) - np.min(filtered_window, axis=0))'''
    
#RMS
def RMS(filtered_window):
    return np.sqrt(np.mean(filtered_window**2, axis=0))