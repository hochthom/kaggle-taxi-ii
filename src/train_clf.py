
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_validation import ShuffleSplit, KFold

from utils import haversineKaggle, rmse, CITY_CENTER



t0 = time.time()
res = {}
for filename in ['train_pp_N1.csv', 'train_pp_N2.csv', 'train_pp_N3.csv', 
                 'train_pp_RND.csv']:
    print('reading training data from %s ...' % filename)
    df = pd.read_csv(os.path.join('../data/', filename), nrows=500000)

    y = np.log(df['len'].values*15 + 1)
    d1 = haversineKaggle(df[['xs', 'ys']].values, df[['xe', 'ye']].values)
    df.drop(['CALL_TYPE', 'TAXI_ID', 'xe', 'ye', 'len'], axis=1, inplace=True)
    X = np.array(df, dtype=np.float)

    # remove long distance
    #d1 = haversineKaggle(CITY_CENTER, y)
    th1 = np.percentile(d1, [99.9])
    
    X = X[(d1<th1), :]
    y = y[(d1<th1)]
    print('Training set size: %i x %i' % X.shape)


    y_pred_rf = np.zeros(y.shape)
    y_pred_gb = np.zeros(y.shape)
    for trn_idx, val_idx in KFold(X.shape[0], n_folds=5):
        # split training data
        X_trn, X_tst, y_trn, y_tst = X[trn_idx,:], X[val_idx,:], y[trn_idx], y[val_idx] 
    
        # Initialize the famous Random Forest Regressor from scikit-learn
        clf = RandomForestRegressor(n_estimators=50, n_jobs=4, random_state=23)
        clf.fit(X_trn, y_trn)
        y_pred_rf[val_idx] = clf.predict(X_tst)
    
        # or the Gradient Boosting Regressor
        clf = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=23)
        clf.fit(X_trn, y_trn)
        y_pred_gb[val_idx] = clf.predict(X_tst)
        
        print('  Score RFR/GBR: %.4f, %.4f' % (rmse(y_tst, y_pred_rf[val_idx]), 
                                               rmse(y_tst, y_pred_gb[val_idx])))


    # save prediction result to file
    err_rf = rmse(y, y_pred_rf)
    err_gb = rmse(y, y_pred_gb)
    id_ = filename.replace('train_pp_','').replace('.csv','')
    res[id_] = {'size':X.shape[0], 'd_th':th1, 'rf':err_rf, 'gb':err_gb}

    print('Total Score: %.4f, %.4f' % (err_rf, err_gb))


with open('training_result_TVT.pkl', 'wb') as fp:
    pickle.dump(res, fp, -1)

print('Done in %.1f sec.' % (time.time() - t0))

