import pandas as pd
import numpy as np
data=pd.read_csv('girish2.csv')
data
features = np.array(data)[:,:-1]
features
target = np.array(data)[:,-1]
target
for i,val in enumerate(target):
if val =='Yes':
specific_h = features[i].copy()
break
print(specific_h)
for i,val in enumerate(features):
if target[i] == 'Yes':
for x in range(len(specific_h)):
if val[x] != specific_h[x]:
specific_h[x] = '?
print(specific_h)
