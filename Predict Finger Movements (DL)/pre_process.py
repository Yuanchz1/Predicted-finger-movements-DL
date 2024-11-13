import scipy.signal as sig
import numpy as np
from scipy.interpolate import CubicSpline

def filter_data(raw_eeg, fs=1000):
    """
    Input:
    raw_eeg (samples x channels): the raw signal
    fs: the sampling rate (1000 for this dataset)
    Output:
    clean_data (samples x channels): the filtered signal
    """
    #Common Average Reference Filtering
    C = raw_eeg.shape[1]
    clean_data = raw_eeg.copy()
    for chn in range(C):
        clean_data[:,chn] = raw_eeg[:,chn]-1/C*np.sum(raw_eeg,axis=1)
    b,a = sig.iirnotch(60, 30, fs=1000)
    clean_data = sig.filtfilt(b,a,clean_data,axis = 0,padlen=10)
    # b2,a2 = sig.iirnotch(120, 30, fs=1000)
    # clean_data = sig.filtfilt(b2,a2,clean_data,axis = 0,padlen=10)
    # b3,a3 = sig.iirnotch(180, 30, fs=1000)
    # clean_data = sig.filtfilt(b3,a3,clean_data,axis = 0,padlen=10)
    return clean_data

def downsample(target_len, y):
    q = int(y.shape[0]/target_len)
    Y = sig.decimate(y, q, axis = 0)
    return Y[:target_len]

def upsample(target_len,y):
    n = y.shape[0]
    x = np.arange(n)
    cs = CubicSpline(x, y)
    xs = np.linspace(0,n,target_len)
    return cs(xs)

def float2int(data):
    return np.rint(data)

def normalize(multichannel_signal, return_values):
    """
    standardization and removal of the median  from each channel
    :param multichannel_signal: Multi-channel signal
    :param return_values: Whether to return standardization parameters. By default - no
    """
    means = np.mean(multichannel_signal, axis=1, keepdims=True)
    stds = np.std(multichannel_signal, axis=1, keepdims=True)
    transformed_data = (multichannel_signal - means) / stds
    common_average = np.median(transformed_data, axis=0, keepdims=True)
    transformed_data = transformed_data - common_average
    
    if return_values:
        return transformed_data, (means, stds)
    return transformed_data

def clean_features(feats):
    bad_feat_inds = np.concatenate((np.arange(2,len(feats.transpose()),7),
                                   np.arange(3,len(feats.transpose()),7),
                                   np.arange(4,len(feats.transpose()),7)))
    
    feats_cleaned = np.delete(feats, bad_feat_inds, axis=1)
    
    return feats_cleaned

def normalize_features(all_feats_train, all_feats_test, num_features):
    # Input should be (num_windows x (channels x features))
    # Num features is the number of unique features that were extracted
    all_feats_train_norm = np.copy(all_feats_train)
    all_feats_test_norm = np.copy(all_feats_test)
    
    for n in range(num_features):
        feats_idx_train = np.arange(n,len(all_feats_train.transpose()),num_features)
        
        feat_data_train = all_feats_train[:][:,feats_idx_train]
        
        feat_means_train = np.mean(feat_data_train,axis=0)
        feat_stds_train = np.std(feat_data_train,axis=0)
        
        all_feats_train_norm[:][:,feats_idx_train] = (feat_data_train - feat_means_train)/feat_stds_train
        
        # Note that we must use the same mean and std. dev from the TRAINING set
        #     because regression models are sensitive to value domain as they are
        #     scale-variant.
        feats_idx_test = np.arange(n,len(all_feats_test.transpose()),num_features)
        feat_data_test = all_feats_test[:][:,feats_idx_test]
        all_feats_test_norm[:][:,feats_idx_test] = (feat_data_test - feat_means_train)/feat_stds_train
        
        # Sanity checking plot, comment out if you don't want plots
#         if n == 0:
#             plt.figure()
#             plt.plot(feat_data_train.transpose()[0])
#             plt.figure()
#             plt.plot(all_feats_train_norm[:][:,feats_idx_train].transpose()[0])
            
#             plt.figure()
#             plt.plot(feat_data_test.transpose()[0])
#             plt.figure()
#             plt.plot(all_feats_test_norm[:][:,feats_idx_test].transpose()[0])

    return all_feats_train_norm, all_feats_test_norm

# Gaussian filtering to clean the output
def convolve_gaussian(preds):
    # preds_t = preds.T()
    preds_convolve = []
    
    fs = 1000
    gaussian_filter = np.exp(-1*(np.arange(int(-1*1000),int(1*1000)))**2/(0.75*1000)**2)
    gaussian_filter_scaled = 1/np.sum(gaussian_filter) * gaussian_filter
    
    for row in preds:
        preds_convolve.append(np.convolve(gaussian_filter_scaled, row, "same"))
    
    # return np.array(preds_convolve).transpose()``
    return np.array(preds_convolve)
    
def convolve_val_gaussian(preds):
    # preds_t = preds.T()
    preds_convolve = []
    
    fs = 1000
    gaussian_filter = 1
    gaussian_filter_scaled = 1/np.sum(gaussian_filter) * gaussian_filter
    
    for row in preds:
        preds_convolve.append(np.convolve(gaussian_filter_scaled, row, "same"))
    
    # return np.array(preds_convolve).transpose()``
    return np.array(preds_convolve)