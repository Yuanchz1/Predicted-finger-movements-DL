from feature_extraction import feature_function
import numpy as np

def get_features(filtered_window, fs=1000):
    """
    Input:
      filtered_window (window_samples x channels): the window of the filtered ecog signal
      fs: sampling rate
    Output:
      features (channels x num_features): the features calculated on each channel for the window
    """
    feature_list = []
    feature_list.append(feature_function.linelength(filtered_window))
    
    feature_list.append(feature_function.time_average(filtered_window))
    
    feature_list.append(feature_function.time_var(filtered_window))
    
    '''feature_list.append(feature_function.bandpower(filtered_window,fs,5,15))
    feature_list.append(feature_function.bandpower(filtered_window,fs,20,25))
    feature_list.append(feature_function.bandpower(filtered_window,fs,75,115))
    feature_list.append(feature_function.bandpower(filtered_window,fs,125,160))
    feature_list.append(feature_function.bandpower(filtered_window,fs,160,175))'''
    
    '''feature_list.append(feature_function.rel_bandpower(filtered_window,fs,1,30))
    feature_list.append(feature_function.rel_bandpower(filtered_window,fs,30,50))
    feature_list.append(feature_function.rel_bandpower(filtered_window,fs,50,80))
    feature_list.append(feature_function.rel_bandpower(filtered_window,fs,80,120))
    feature_list.append(feature_function.rel_bandpower(filtered_window,fs,120,200))'''
    
    feature_list.append(feature_function.rel_bandpower(filtered_window,fs,5,15))
    feature_list.append(feature_function.rel_bandpower(filtered_window,fs,20,25))
    feature_list.append(feature_function.rel_bandpower(filtered_window,fs,75,115))
    feature_list.append(feature_function.rel_bandpower(filtered_window,fs,125,160))
    feature_list.append(feature_function.rel_bandpower(filtered_window,fs,160,175))
    
    feature_list.append(feature_function.total_power(filtered_window,fs))
    
    feature_list.append(feature_function.filt_amp(filtered_window,fs,1,30))
    feature_list.append(feature_function.filt_amp(filtered_window,fs,30,50))
    feature_list.append(feature_function.filt_amp(filtered_window,fs,50,80))
    feature_list.append(feature_function.filt_amp(filtered_window,fs,80,120))
    feature_list.append(feature_function.filt_amp(filtered_window,fs,120,200))
    
    feature_list.append(feature_function.area(filtered_window))
    
    # feature_list.append(feature_function.RMS(filtered_window))
    
    # feature_list.append(feature_function.avg_peak_to_peak(filtered_window))
    
    # feature_list.append(feature_function.energy(filtered_window))
    
    # feature_list.append(feature_function.katz_fractal_dimension(filtered_window))
    
    # feature_list.append(feature_function.ZeroCrossing(filtered_window))
     
    '''feature_list.append(feature_function.freq_average(filtered_window,fs,5,15))
    feature_list.append(feature_function.freq_average(filtered_window,fs,20,25))
    feature_list.append(feature_function.freq_average(filtered_window,fs,75,115))
    feature_list.append(feature_function.freq_average(filtered_window,fs,125,160))
    feature_list.append(feature_function.freq_average(filtered_window,fs,160,175))'''
    
    '''feature_list.append(feature_function.freq_average(filtered_window,fs,1,30))
    feature_list.append(feature_function.freq_average(filtered_window,fs,30,50))
    feature_list.append(feature_function.freq_average(filtered_window,fs,50,80))
    feature_list.append(feature_function.freq_average(filtered_window,fs,80,120))
    feature_list.append(feature_function.freq_average(filtered_window,fs,120,200))'''

    return np.vstack(feature_list).T
