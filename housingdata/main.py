import pandas as pd
import numpy as np

fname = 'train.csv'
data = pd.read_csv(fname)
len(data)
data.head()

data.count()
data['indus'].value_counts()
data['chas'].value_counts()
data['crim'].value_counts()
data['zn'].value_counts()
data['nox'].value_counts()
data['rm'].value_counts()
data['age'].value_counts()
data['dis'].value_counts()
data['rad'].value_counts()
data['tax'].value_counts()
data['ptratio'].value_counts()
data['black'].value_counts()
data['lstat'].value_counts()
data['medv'].value_counts()

columns_target= ['medv']
columns_train = ['ID','crim','indus','nox','rm','age','dis','rad','tax','ptratio','black','lstat']

X= data[columns_train]
Y= data[columns_target]

X['lstat'].isnull().sum()
print(Y.head())

from sklearn import linear_model    
clf=linear_model.LinearRegression()
clf.fit(X, Y)
test = pd.read_csv('test.csv')
Z = test[columns_train]
print(Z.head())
pred = clf.predict(Z)
print pred
for i in pred:
    print(i)
np.savetxt('solution.csv',pred,delimiter=' ')

