
import time
import json
import numpy as np
import pandas as pd

from utils import haversineKaggle, heading, CITY_CENTER


def process_row_training(row):
    x = row['POLYLINE']
    if len(x)>4:
        x = np.array(x, ndmin=2)
        data = process_trip(x[0, :], row['TIMESTAMP'])
        data += [x[-1,0], x[-1,1], len(x)]
    else:
        data = [-1]*9
    return pd.Series(np.array(data, dtype=float))

def process_row_test(row):
    x = row['POLYLINE']
    x = np.array(x, ndmin=2)
    data = process_trip(x[0, :], row['TIMESTAMP'])
    return pd.Series(np.array(data, dtype=float))

def process_trip(x, start_time):
    tt = time.localtime(start_time)
    data = [tt.tm_wday, tt.tm_hour]
    # distance from the center till cutting point
    d_st = haversineKaggle(x,  CITY_CENTER)
    head = heading(x,  CITY_CENTER[0])
    data += [x[0], x[1], d_st, head]
    return data


t0 = time.time()
FEATURES = ['wday','hour','xs','ys','d_st','heading']

            
print('reading training data ...')
df = pd.read_csv('../data/train.csv', converters={'POLYLINE': lambda x: json.loads(x)})#, nrows=100)

print('preparing train data ...')
ds = df.apply(process_row_training, axis=1)
ds.columns = FEATURES + ['xe','ye','len']
df.drop(['POLYLINE','TIMESTAMP','TRIP_ID','DAY_TYPE','ORIGIN_CALL','ORIGIN_STAND'], 
        axis=1, inplace=True)
df['TAXI_ID'] -= np.min(df['TAXI_ID'])   # makes csv smaller -> ids in [0, 980]
df = df.join(ds)

# clean up tracks
df = df[(df['xe'] != -1) & (df['MISSING_DATA']==False)]
df.drop(['MISSING_DATA'], axis=1, inplace=True)
df.to_csv('../data/train_pp_N1.csv', index=False)


print('reading test data ...')
df = pd.read_csv('../data/test.csv', converters={'POLYLINE': lambda x: json.loads(x)})

print('preparing test data ...')
ds = df.apply(process_row_test, axis=1)
ds.columns = FEATURES
df.drop(['POLYLINE','TIMESTAMP','DAY_TYPE','ORIGIN_CALL','ORIGIN_STAND',
         'MISSING_DATA'], axis=1, inplace=True)
df = df.join(ds)
df.to_csv('../data/test_pp_N1.csv', index=False)

print('Done in %.1f sec.' % (time.time()-t0))

