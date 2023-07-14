import pandas as pd
import numpy as np
df=pd.read_csv("emp.csv")
df
df.head()
df.describe()
df.info()
df.isnull().sum()
df.drop("name",axis=1,inplace=True)
df
df.drop(10,axis=0,inplace=True)
df
df["age"]=df["age"].fillna(df["age"].mean())
df
df["salary"]=df["salary"].fillna(df["salary"].max())
df
df.drop_duplicates()
x=np.array(df)[:,:-1]
x
y=np.array(df)[:,-1]
y
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder="passthrough")
x=np.array(ct.fit_transform(x))
x
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform()
y
from sklearn.model_selection import train_test_split
x_test
y_train
y_test
