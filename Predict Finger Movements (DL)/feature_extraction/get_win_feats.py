import numpy as np
import pre_process
import feature_extraction.get_feature as get_feature
def NumWins(x,fs,winLen,winOver): 
  xLen = x.shape[0]
  sample_cover = winLen*fs*0.001
  sample_disp = winOver*fs*0.001
  return int((xLen-sample_cover)//(sample_cover-sample_disp)+1)

def get_windowed_feats(raw_ecog, fs, window_length, window_overlap):
    '''
    Inputs:
      raw_eeg (samples x channels): the raw signal
      fs: the sampling rate (1000 for this dataset)
      window_length: the window's length
      window_overlap: the window's overlap
    Output:
      all_feats (num_windows x channels x features): the features for each channel for each time window
    '''
    feature_list = []
    num_win = NumWins(raw_ecog,fs,window_length,window_overlap)
    for i in range(num_win):
        win_temp = raw_ecog[:,:][int(i*(window_length-window_overlap)*fs/1000):int(i*(window_length-window_overlap)*fs/1000+window_length*fs/1000)]
        filtered_win = pre_process.filter_data(win_temp,fs=1000)
        feature = get_feature.get_features(filtered_win,fs =1000)
        # feature = get_feature.get_features(win_temp,fs =1000)
        # feature_list.append(feature)
        feature_list.append(feature.reshape(1,-1))
        
    # return np.stack(feature_list)
    return np.vstack(feature_list)