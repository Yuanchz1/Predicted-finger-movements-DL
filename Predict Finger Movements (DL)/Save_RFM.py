from sklearn.ensemble import RandomForestRegressor
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
#test_x = load_data.load_test_data("leaderboard_data.mat")

# deleting invalid channels
train_x[0] = np.delete(train_x[0],[54],1)
#test_x[0][0] = np.delete(test_x[0][0],[54],1)

train_x[1] = np.delete(train_x[1],[20,37],1)
#test_x[1][0] = np.delete(test_x[1][0],[20,37],1)

# sigma_list = [6,4,3]
test_x, test_y = loadmat("new_data.mat")
test_x[0][0] = np.delete(test_x[0][0],[54],1)
test_x[1][0] = np.delete(test_x[1][0],[20,37],1)
predictions = np.zeros((3,1), dtype=object)
regressor = []
for i in range(3):
    #pred_temp = []
    scaler = StandardScaler()
    reg = RandomForestRegressor(n_estimators=100, n_jobs = -1)  
    train_X, train_Y = pre_process.filter_data(train_x[i]), train_y[i]  # Selecting data for each finger
    # train_X = scaler.fit_transform(train_X)
    #test_X = pre_process.filter_data(test_x[i][0]) # Selecting data for each finger
    #test_X = scaler.transform(test_X)
    train_y_down = pre_process.downsample(int(25 * train_Y.shape[0] / 1000), train_Y)
    feature_train = get_win_feats.get_windowed_feats(train_X, 1000, 80, 40)
    #feature_test = get_win_feats.get_windowed_feats(test_X, 1000, 80, 40)
    reg.fit(feature_train, pre_process.downsample(feature_train.shape[0], train_y_down))
    
    regressor.append(reg)
        
### Save final predictions to 'prediction.mat' with one variable called 'predicted_dg'
#savemat('prediction.mat',{'predicted_dg':predictions})
'''
models = {
    's1f1':regressor[0],
    's1f2':regressor[1],
    's1f3':regressor[2],
}
with open("RF_Models.pkl", "wb") as f:
    pickle.dump(models, f)'''



