
import os
import time
import numpy as np
import cPickle as pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_validation import ShuffleSplit, KFold

from utils import haversineKaggle, rmse, CITY_CENTER


   
t0 = time.time()

res = {}
for id_ in range(320):
    filename = '../data/train_pp_TST_%i.csv' % id_
    if not os.path.isfile(filename):
        continue
    
    df = pd.read_csv(filename)
    if df.shape[0] < 1000:
        #print('skipping key point %i (%i)' % (id_, df.shape[0]))
        continue
    
    y = np.log(df['len'].values*15 + 1)
    d1 = haversineKaggle(df[['xs', 'ys']].values, df[['xe', 'ye']].values)
    df.drop(['CALL_TYPE', 'TAXI_ID', 'xe', 'ye', 'len'], axis=1, inplace=True)
    X = np.array(df, dtype=np.float)    
    
    # remove long distance
    #d1 = haversineKaggle(CITY_CENTER, y)
    th1 = np.percentile(d1, [99.9])
   
    X = X[(d1<th1), :]
    y = y[(d1<th1)]
    print('- processing id %3i (%i)' % (id_, X.shape[0]))
    
    y_pred_rf = np.zeros(y.shape)
    y_pred_gb = np.zeros(y.shape)
    for trn_idx, val_idx in KFold(X.shape[0], n_folds=5):
    #for trn_idx, val_idx in ShuffleSplit(X.shape[0], n_iter=1, test_size=0.2, random_state=42):
        # split training data
        X_trn, X_tst, y_trn, y_tst = X[trn_idx,:], X[val_idx,:], y[trn_idx], y[val_idx] 
    
        # Initialize the famous Random Forest Regressor from scikit-learn
        clf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=23)
        clf.fit(X_trn, y_trn)
        y_pred_rf[val_idx] = clf.predict(X_tst)
    
        # or the Gradient Boosting Regressor
        clf = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=23)
        clf.fit(X_trn, y_trn)
        y_pred_gb[val_idx] = clf.predict(X_tst)
        
        print('  Score RFR/GBR: %.4f, %.4f' % (rmse(y_tst, y_pred_rf[val_idx]), 
                                               rmse(y_tst, y_pred_gb[val_idx])))
                
    
    err_rf = rmse(y, y_pred_rf)
    err_gb = rmse(y, y_pred_gb)
    res[id_] = {'size':X.shape[0], 'd_th':th1, 'rf':err_rf, 'gb':err_gb}
    print('Total Score: %.4f, %.4f' % (err_rf, err_gb))


with open('training_result_TVT_experts.pkl', 'wb') as fp:
    pickle.dump(res, fp, -1)

print('Done in %.1f sec.' % (time.time() - t0))

