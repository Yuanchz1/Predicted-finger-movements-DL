from sklearn.linear_model import LinearRegression
import pre_process
import load_data
from feature_extraction import get_win_feats
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from load_data import load_full_data, load_test_data
import numpy as np

val, train = load_data.split_train_val("raw_training_data.mat", 0.2)

# scaler = MinMaxScaler(feature_range=(-5, 5))
predictions = np.zeros((3, 1), dtype=object)

val[0][0] = np.delete(val[0][0], [54], 1)
train[0][0] = np.delete(train[0][0], [54], 1)

val[0][1] = np.delete(val[0][1], [20, 37], 1)
train[0][1] = np.delete(train[0][1], [20, 37], 1)
# test_X[1][0] = np.delete(test_X[1][0],[20,37],1)

mean_corr = []
for i in range(3):
    scaler = StandardScaler()
    print("subject {}".format(i + 1))
    reg = LinearRegression()  # Change here to LinearRegression
    train_x, train_y = pre_process.filter_data(train[0][i]), train[1][i]
    train_x = scaler.fit_transform(train_x)
    val_x, val_y = pre_process.filter_data(val[0][i]), val[1][i]
    val_x = scaler.transform(val_x)
    train_y_down = pre_process.downsample(int(25 * train_y.shape[0] / 1000), train_y)
    val_y_down = pre_process.downsample(int(25 * val_y.shape[0] / 1000), val_y)

    feature_train = get_win_feats.get_windowed_feats(train_x, 1000, 80, 40)
    feature_val = get_win_feats.get_windowed_feats(val_x, 1000, 80, 40)
    reg.fit(feature_train, pre_process.downsample(feature_train.shape[0], train_y_down))
    pred = reg.predict(feature_val)
    # scaler = MinMaxScaler((-1,5))
    scaler = MinMaxScaler((-1, 2))
    pred_ = scaler.fit_transform(gaussian_filter1d(pred, sigma=4, axis=0))
    # thre = 1
    thre = 0
    pred__ = pred_.copy()
    for m in range(pred__.shape[0]):
        for n in range(pred__.shape[1]):
            if pred__[m, n] > thre:
                # pred__[m,n]*=2.5
                pred__[m, n] *= 2
    corr = []
    for j in range(5):
        corr.append(pearsonr(pre_process.upsample(val_y.shape[0], pred__)[:, j], val_y[:, j])[0])
        print('Finger {}'.format(j + 1), corr[j])
    print('---------------------------------------------')
    mean_corr.append(np.mean(corr))
    print(np.mean(corr))

print('overall mean:', np.mean(mean_corr))
