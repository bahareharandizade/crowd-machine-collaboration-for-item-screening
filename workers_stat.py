import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats

n_papers = 100
n_criteria = 3


data = pd.read_csv('output/amt_data/crowd-data.csv')
workers_ids_ch = set(list(pd.unique(data['intervention worker ID'])) \
                               + list(pd.unique(data['use of tech worker ID'])) \
                               + list(pd.unique(data['older adult worker ID'])))
# workers_ids_ch = set(list(pd.unique(data['use of tech worker ID'])) \
#                      + list(pd.unique(data['older adult worker ID'])))
workers_ids_ch = [id_ch for id_ch in workers_ids_ch if type(id_ch) != float]
workers_data = [[] for _ in workers_ids_ch]

paper_ids_dict = dict(zip(set(data['paper ID']), range(n_papers)))
# property of the data file
criteria = {0: 'intervention Vote',
            1: 'use of tech vote',
            2: 'older adult vote'}
column_workers_ids = ['intervention worker ID', 'use of tech worker ID', 'older adult worker ID']
# criteria = {
#     0: 'use of tech vote',
#     1: 'older adult vote'}
# column_workers_ids = ['use of tech worker ID', 'older adult worker ID']
column_gt = ['is intervention vote accurate?', 'is tech vote accurate?', 'is older vote accurate?']

data_df = []
for w_id, w_ch in enumerate(workers_ids_ch):
    for c_id, c_name in criteria.iteritems():
        column_workers = column_workers_ids[c_id]
        column_g = column_gt[c_id]
        c_data = data[data[column_workers] == w_ch][['paper ID', column_workers, c_name, column_g]]
        if len(c_data) < 5:
            continue
        w_acc = sum(c_data[column_g]) / len(c_data)
        if w_acc >= 0.7:
            data_df.append([w_ch, c_name, w_acc, len(c_data)])
pd.DataFrame(data_df, columns=['worker_id', 'criteria', 'accuracy', 'n_votes']).to_csv('output/amt_data/workers_stat2.csv', index=False)