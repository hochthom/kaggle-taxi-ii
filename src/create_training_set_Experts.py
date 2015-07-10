
import time
import json
import numpy as np
import pandas as pd

from utils import haversineKaggle, heading, CITY_CENTER, LON_SCALE, LAT_SCALE


def process_row_training(X, row, end_pts):
    pln = row['POLYLINE']
    if len(pln) > 4:
        pln = np.array(pln, ndmin=2)
        
        for id_, pos in end_pts.iteritems():
            # calc dist to end point
            d1 = np.abs(pln[:, 0] - pos[0])*LON_SCALE + \
                 np.abs(pln[:, 1] - pos[1])*LAT_SCALE

            if np.min(d1) < 0.05: #e.g. km
                idx = np.argmin(d1)
                data = [row['CALL_TYPE'], row['TAXI_ID']]
                data += process_trip(pln[:idx+1, :], row['TIMESTAMP'])
                data += [pln[-1,0], pln[-1,1], len(pln)]
            
                X.setdefault(id_, []).append(data)
    return X

def process_row_test(row):
    x = row['POLYLINE']
    x = np.array(x, ndmin=2)
    data = process_trip(x, row['TIMESTAMP'])
    return pd.Series(np.array(data, dtype=float))

def process_trip(x, start_time):
    tt = time.localtime(start_time)
    data = [tt.tm_wday, tt.tm_hour]
    # cumulative sum of distance
    d_cs = 0
    vcar = 0
    vmed = 0
    head = 0
    if x.shape[0] > 1:
        d1 = haversineKaggle(x[:-1,:], x[1:,:])
        d_cs = np.sum(d1)
        vmed = np.median(d1)
        vcar = d1[-1]
        head = heading(x[-2,:], x[-1,:])
    # distance from the center till cutting point
    d_st = haversineKaggle(x[0,:],  CITY_CENTER)[0]
    h_st = heading(x[0,:],  CITY_CENTER[0])
    d_cut = haversineKaggle(x[-1,:], CITY_CENTER)[0]
    h_cut = heading(CITY_CENTER[0], x[-1,:])
    data += [x.shape[0], x[0,0], x[0,1], x[-1,0], x[-1,1], d_st, h_st, d_cut, 
             h_cut, d_cs, vmed, vcar, head]
    return data


FEATURES = ['wday','hour','length','xs','ys','x1','y1','d_st','h_st',
            'd_cut','h_cut','d_cs','vmed','vcar','heading']


t0 = time.time()
print('reading test data ...')
df = pd.read_csv('../data/test.csv', converters={'POLYLINE': lambda x: json.loads(x)})#, nrows=10000)

ds = df.apply(process_row_test, axis=1)
ds.columns = FEATURES
df = df.join(ds)
end_pts = dict((i, x) for i, x in enumerate(df[['x1','y1']].values))


print('reading training data ...')
df = pd.read_csv('../data/train.csv', converters={'POLYLINE': lambda x: json.loads(x)})#, nrows=50000)

print ('preparing train data ...')
X = {}
for i in range(df.shape[0]):
    X = process_row_training(X, df.iloc[i], end_pts)
    
del df
for id_, data in X.iteritems():    
    df = pd.DataFrame(data, columns = ['CALL_TYPE','TAXI_ID'] + FEATURES + ['xe','ye','len'])
    df['TAXI_ID'] -= np.min(df['TAXI_ID'])   # makes csv smaller -> ids in [0, 980]
    df.to_csv('../data/train_pp_TST_%i.csv' % (id_), index=False)

print('Done in %.1f sec.' % (time.time() - t0))

