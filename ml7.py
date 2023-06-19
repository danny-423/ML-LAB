import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import metrics

data=pd.read_csv('textdata.csv',names=['message','label'])
print('The dataset is\n',data)
print('The dimensions of the dataset',data.shape)
data['labelnum']=data.label.map({'pos':1,'neg':0})
X=data.message
y=data.labelnum
print(X)
print(y)
vectorizer=TfidfVectorizer()
data=vectorizer.fit_transform(X)
print('\n the Features of dataset:\n')
df=pd.DataFrame(data.toarray(),columns=vectorizer.get_feature_names_out())
df.head()
print('\n Train test split')
xtrain,xtest,ytrain,ytest=train_test_split(data,y,test_size=0.3,random_state=42)
print('\n the total number of training data:',ytrain.shape)
print('\n the total number of test data:',ytest.shape)
clf=MultinomialNB().fit(xtrain,ytrain)
predict=clf.predict(xtest)
predicted=clf.predict(xtest)
print('\n Accuracy of the classifier is',metrics.accuracy_score(ytest,predicted))
print('\n Confusion Matrix is\n',metrics.confusion_matrix(ytest,predicted))
print('\n Classification report is\n',metrics.classification_report(ytest,predicted))
print('\n value of precision is',metrics.precision_score(ytest,predicted))
print('\n value of recall is',metrics.recall_score(ytest,predicted))