import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

datasets = pd.read_csv('Downloads/10.csv')
X = datasets.iloc[:, [2,3]].values
Y = datasets.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_Train, Y_Train)
Y_Pred = classifier.predict(X_Test)

from sklearn import metrics
print("Accuracy score", metrics.accuracy_score(Y_Test, Y_Pred))

plt.scatter(X_Train[:,0],X_Train[:, 1], c=Y_Train)

plt.scatter(X_Train[:,0], X_Train[:,1], c= Y_Train)
plt.title('Support Vector machine (Traning Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
w = classifier.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-2.5,2.5)
yy = a*xx - (classifier.intercept_[0])/w[1]
plt.plot(xx,yy)
plt.show()
plt.scatter(X_Test[:,0],X_Test[:, 1],c=Y_Test)
plt.title('Support Vector Machine (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
w = classifier.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-2.5,2.5)
yy = a*xx - (classifier.intercept_[0])/w[1]
plt.plot(xx,yy)
plt.show()
