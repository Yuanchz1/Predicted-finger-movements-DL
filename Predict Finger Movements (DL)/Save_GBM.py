from sklearn.ensemble import GradientBoostingRegressor
import pre_process
import load_data
from feature_extraction import get_win_feats
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from load_data import load_full_data, load_test_data
import numpy as np
from scipy.io import loadmat
import pickle

train_x, train_y = load_data.load_full_data("raw_training_data.mat")

# deleting invalid channels
train_x[0] = np.delete(train_x[0],[54],1)

train_x[1] = np.delete(train_x[1],[20,37],1)

predictions = np.zeros((3,1), dtype=object)
regressor = []
for i in range(3):
    scaler = MinMaxScaler((-1,2))
    train_X = pre_process.filter_data(train_x[i])  # Selecting data for each finger
    train_X = scaler.fit_transform(train_X)
    
    for j in range(5):  # Loop through each finger
        train_Y = train_y[i][:, j]
        # reg = GradientBoostingRegressor(n_estimators=100, random_state = 2) 
        reg = GradientBoostingRegressor(n_estimators=100)  
        train_y_down = pre_process.downsample(int(25 * train_Y.shape[0] / 1000), train_Y)
        feature_train = get_win_feats.get_windowed_feats(train_X, 1000, 80, 40)
        reg.fit(feature_train, pre_process.downsample(feature_train.shape[0], train_y_down))
        regressor.append(reg)
        
### Save final predictions to 'prediction.mat' with one variable called 'predicted_dg'
# savemat('prediction.mat',{'predicted_dg':predictions})
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
with open("GB_Models.pkl", "wb") as f:
    pickle.dump(models, f)