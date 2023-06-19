import pandas as pd
col = ['Age', 'Gender', 'FamilyHist', 'Diet', 'LifeStyle', 'Cholestrol', 'HeartDisease']

data = pd.read_csv("Desktop\lab8.csv",names = col)
print(data)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for i in range(len(col)):
    data.iloc[:,i] = encoder.fit_transform(data.iloc[:,i])

x = data.iloc[:,0:6]
y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

from sklearn.metrics import confusion_matrix
print('confusion matrix',confusion_matrix(y_test, y_pred))