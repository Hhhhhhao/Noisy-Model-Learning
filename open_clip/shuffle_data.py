import os
from glob import glob
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
import webdataset as wds
import random
from copy import deepcopy

noise_ratio = 0.025
data_json = '/media/Auriga/datasets/yfcc15m/yfcc15m_clean_open_data.json'
save_json = '/media/Auriga/datasets/yfcc15m/yfcc15m_clean_open_data_noise_{}.json'.format(noise_ratio * 100)


data = pd.read_json(data_json, lines=True)
# new_data = pd.read_json(save_json, lines=True)

# cnt = 0
# for i in range(len(data)):
#     if data.iloc[i]['caption'] != new_data.iloc[i]['caption']:
#         cnt+=1 
# print(cnt)
# print(len(data))
noise_samples = int(len(data) * noise_ratio)
pool = np.arange(len(data)).tolist()

for _ in tqdm(range(noise_samples)):
    idx1, idx2 = random.sample(pool, 2)
    tmp = deepcopy(data.iloc[idx1]['caption'])
    data.iloc[idx1]['caption'] =  deepcopy(data.iloc[idx2]['caption'])
    data.iloc[idx2]['caption'] = tmp

data.to_json(save_json, orient="records", lines=True)
