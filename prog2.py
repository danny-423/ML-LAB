import pandas as pd
import numpy as np
data=pd.read_csv('girish1.csv')
data
features=np.array(data)[:,:-1]
features
target=np.array(data)[:,-1]
target
for i, h in enumerate(features):
if target[i] =='yes':
specific_h=features[i].copy()
break
print("initialization of specific_h and general_h")
print(specific_h)
general_h=[["?" for i in range (len(specific_h))] for i in range (len(specific_h))]
print(general_h)
for i, h in enumerate(features):
if target[i] =='yes':
for x in range(len(specific_h)):
if h[x]!=specific_h[x]:
specific_h[x]='?'
general_h[x][x]='?'
if target[i] =='no':
for x in range (len(specific_h)):
if h[x]!=specific_h[x]:
general_h[x][x]=specific_h[x]
else:
general_h[x][x]='?'
print(specific_h,"\n")
print(general_h,"\n")
indices=[i for i,val in enumerate(general_h) if val==['?', '?', '?', '?', '?', '?']]
for i in indices:
general_h.remove(['?', '?', '?', '?', '?', '?'])
print("Final Specific_h:",specific_h , sep="\n")
print("Final General_h:", general_h, sep="\n")
