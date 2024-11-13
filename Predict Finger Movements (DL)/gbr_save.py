import numpy as np
from model import linear_regression as LR
from model import RandomForest as RF
from load_data import load_full_data,load_test_data
from pre_process import upsample,downsample
from scipy.io import savemat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pre_process, load_data
from feature_extraction import get_win_feats
import pickle

train_x, train_y = load_data.load_full_data("raw_training_data.mat")
#test_x = load_data.load_test_data("leaderboard_data.mat")

# deleting invalid channels
train_x[0] = np.delete(train_x[0],[54],1)
#test_x[0][0] = np.delete(test_x[0][0],[54],1)

train_x[1] = np.delete(train_x[1],[20,37],1)
#test_x[1][0] = np.delete(test_x[1][0],[20,37],1)

predictions = np.zeros((3,1), dtype=object)
regressor = []
for i in range(3):
    #pred_temp = []
    scaler = StandardScaler()
    for j in range(5):  # Loop through each finger
        reg = GradientBoostingRegressor(n_estimators=100)  
        train_X, train_Y = pre_process.filter_data(train_x[i]), train_y[i][:, j]  # Selecting data for each finger
        train_X = scaler.fit_transform(train_X)
        #test_X = pre_process.filter_data(test_x[i][0]) # Selecting data for each finger
        #test_X = scaler.transform(test_X)
        train_y_down = pre_process.downsample(int(25 * train_Y.shape[0] / 1000), train_Y)
        feature_train = get_win_feats.get_windowed_feats(train_X, 1000, 80, 40)
        #feature_test = get_win_feats.get_windowed_feats(test_X, 1000, 80, 40)
        reg.fit(feature_train, pre_process.downsample(feature_train.shape[0], train_y_down))
        regressor.append(reg)
        '''
        pred = reg.predict(feature_test)
        # scaler = MinMaxScaler((-1,5))
        scaler = MinMaxScaler((-1, 2))
        pred = pred.reshape(-1, 1)
        pred_ = scaler.fit_transform(gaussian_filter1d(pred, sigma=4, axis=0))
        # thre = 1
        thre = 0
        pred__ = pred_.copy()
        for m in range(pred__.shape[0]):
            for n in range(pred__.shape[1]):
                if pred__[m, n] > thre:
                    # pred__[m,n]*=2.5
                    pred__[m, n] *= 2   
        pred_temp.append(pre_process.upsample(test_X.shape[0],pred__).reshape(-1))
        ### Generate predictions for channel #1 and #2
    predictions[i,0] = np.vstack(pred_temp).T
    '''
### Save final predictions to 'prediction.mat' with one variable called 'predicted_dg'
#savemat('prediction.mat',{'predicted_dg':predictions})
models = {
    's1f1':regressor[0],
    's1f2':regressor[1],
    's1f3':regressor[2],
    's1f4':regressor[3],
    's1f5':regressor[4],
    's2f1':regressor[5],
    's2f2':regressor[6],
    's2f3':regressor[7],
    's2f4':regressor[8],
    's2f5':regressor[9],
    's3f1':regressor[10],
    's3f2':regressor[11],
    's3f3':regressor[12],
    's3f4':regressor[13],
    's3f5':regressor[14],
}
with open("models.pkl", "wb") as f:
    pickle.dump(models, f)



