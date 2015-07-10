
import os
import time
import numpy as np
import cPickle as pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_validation import ShuffleSplit

from utils import haversineKaggle, rmse, CITY_CENTER

   
t0 = time.time()
df = pd.read_csv('../data/test_pp_RND.csv')
df = df.drop(['TRIP_ID', 'CALL_TYPE', 'TAXI_ID'], axis = 1)
X_tst = np.array(df, dtype=np.float)

pred = {}
for id_ in range(320):
    filename = '../data/train_pp_TST_%i.csv' % id_
    if not os.path.isfile(filename):
        continue
    
    df = pd.read_csv(filename)
    if df.shape[0] < 1000:
        print('skipping key point %i (%i)' % (id_, df.shape[0]))
        continue
    
    # factorize categorical columns in training set
    #df['CALL_TYPE'], ct_index = pd.factorize(df['CALL_TYPE'])
    #df = df[df['CALL_TYPE'] == 0]    # A=2, B=1, C=0
    # fill all NaN values with -1
    #df = df.fillna(-1)
        
    # remove long distance
    d1 = haversineKaggle(df[['xs', 'ys']], df[['xe', 'ye']])
    th1 = np.percentile(d1, [99.9])
    df = df.loc[d1 < th1]

    y = np.ravel(np.log(df['len']*15 + 1))
    df.drop(['CALL_TYPE', 'TAXI_ID', 'xe', 'ye', 'len'], axis=1, inplace=True)
    X = np.array(df, dtype=np.float)

    print('training classifier of key point %i  (sz=%i) ...' % (id_, X.shape[0]))                                            
    # Initialize the famous Random Forest Regressor from scikit-learn
    clf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=21)
    clf.fit(X, y)
    pred_rf = clf.predict(X_tst[id_, :])

    clf = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=21)
    clf.fit(X, y)
    pred_gb = clf.predict(X_tst[id_, :])
        
    #print 'predicting test data ...'
    pred[id_] = {'rfr':pred_rf, 'gbr':pred_gb, 'size':X.shape[0]}


with open('predictions_TVT_experts.pkl', 'wb') as fp:
    pickle.dump(pred, fp, -1)

print('Done in %.1f sec.' % (time.time() - t0))

