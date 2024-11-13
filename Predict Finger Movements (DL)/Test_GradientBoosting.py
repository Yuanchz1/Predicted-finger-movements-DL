from sklearn.ensemble import GradientBoostingRegressor
import pre_process
from feature_extraction import get_win_feats
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from load_data import load_full_data, load_test_data
import numpy as np
from scipy.io import loadmat


train_X, train_Y = load_full_data("raw_training_data.mat")

val_X = loadmat("new_data.mat")["ecog"]
val_Y = loadmat("new_data.mat")["dg"]
predictions = np.zeros((3, 1), dtype=object)

val_X[0][0] = np.delete(val_X[0][0], [54], 1)
train_X[0] = np.delete(train_X[0], [54], 1)

val_X[1][0] = np.delete(val_X[1][0], [20, 37], 1)
train_X[1] = np.delete(train_X[1], [20, 37], 1)

sigma_list = [10.75, 4, 7.75]
factor_list = [30, 6.5, 4]

mean_corr = []
for i in range(3):
    scaler = StandardScaler()
    print("subject {}".format(i + 1))
    mean_corr_subject = []
    for j in range(5):  # Loop through each finger
        reg = GradientBoostingRegressor(n_estimators=100)  # Change here
        train_x, train_y = pre_process.filter_data(train_X[i]), train_Y[i][:, j]  # Selecting data for each finger
        train_x = scaler.fit_transform(train_x)
        val_x, val_y = pre_process.filter_data(val_X[i][0]), val_Y[i][0][:, j]  # Selecting data for each finger
        val_x = scaler.transform(val_x)
        train_y_down = pre_process.downsample(int(25 * train_y.shape[0] / 1000), train_y)
        val_y_down = pre_process.downsample(int(25 * val_y.shape[0] / 1000), val_y)

        feature_train = get_win_feats.get_windowed_feats(train_x, 1000, 80, 40)
        feature_val = get_win_feats.get_windowed_feats(val_x, 1000, 80, 40)
        reg.fit(feature_train, pre_process.downsample(feature_train.shape[0], train_y_down))
        pred = reg.predict(feature_val)
       
        scaler = MinMaxScaler((-1, 2))
        pred = pred.reshape(-1, 1)
        
        scaler1 = StandardScaler()
        
        scaler2 = RobustScaler(unit_variance=True, quantile_range=(25, 75))
        
        scaler3 = MinMaxScaler((-1, 2))
        
        scaler_list = [scaler1, scaler2, scaler3]
        
        pred_ = scaler_list.fit_transform(gaussian_filter1d(pred, sigma=sigma_list[i], axis=0))
        
        thre = 0
        pred__ = pred_.copy()
        for m in range(pred__.shape[0]):
            for n in range(pred__.shape[1]):
                if pred__[m, n] > thre:
                    pred__[m,n] *= factor_list
                    
        corr = pearsonr(pre_process.upsample(val_y.shape[0], pred__).reshape(-1), val_y)[0]
        print('Finger {}:'.format(j + 1), corr)
        mean_corr_subject.append(corr)
        
    print('---------------------------------------------')
    mean_corr.append(np.mean(mean_corr_subject))
    print("Mean correlation for subject {}:".format(i + 1), np.mean(mean_corr_subject))

print('Overall mean correlation:', np.mean(mean_corr))