
import numpy as np
import pandas as pd
import cPickle as pickle

            

print('reading general model predictions ...')
df = pd.read_csv('../data/test_pp_RND.csv')

submissions = ['my_submission_N1.csv','my_submission_N2.csv',
               'my_submission_N3.csv','my_submission_RND.csv']


idx = np.argsort(df['length'])
for i, fn in enumerate(submissions): 
    tmp = pd.read_csv(fn)
    df['F%i' % i] = tmp['TRAVEL_TIME']


print('creating submission 1 ...')
df['TRAVEL_TIME'] = df['F3']
idx = np.where(df['length']<=15)[0]
df.loc[idx, 'TRAVEL_TIME'] = np.mean(df.loc[idx, ['F0','F1','F2','F3']], axis=1)

submission = df[['TRIP_ID','TRAVEL_TIME']]
submission.to_csv('final_submission_1.csv', index = False)


print 'reading trip specific expert predictions ...'
with open('predictions_TVT_experts.pkl', 'rb') as fp:
    pred = pickle.load(fp)

print('creating submission 2 ...')
for id_, res in pred.iteritems():
    if res['size'] > 1000:
        df.loc[id_, 'TRAVEL_TIME'] = np.exp(0.5*(res['rfr']+res['gbr']))    

submission = df[['TRIP_ID','TRAVEL_TIME']]
submission.to_csv('final_submission_2.csv', index = False)
print('Done.')

