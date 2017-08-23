import pandas as pd
import numpy as np

fname = 'xtrain.csv'
data = pd.read_csv(fname)
len(data)
data.head()

data.count()

data['age'].min(),data['age'].max()
data['survived'].value_counts()
data['sex'].value_counts()

columns_target= ['survived']
columns_train = ['age','pclass','sex','fare']

X= data[columns_train]
Y= data[columns_target]

X['sex'].isnull().sum()
X['pclass'].isnull().sum()
X['fare'].isnull().sum()	
X['age'].isnull().sum()
X['age'] = X['age'].fillna(X['age'].median())
X['age'].isnull().sum()
X['fare'] = X['fare'].fillna(X['fare'].median())
X['fare'].isnull().sum()
X['age'].isnull().sum()
X['fare'].isnull().sum()
d={'male':0,'female':1}	
data.convert_objects(convert_numeric=True)
data.fillna(0, inplace=True)
def handle_non_numerical_data(X):
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if X['sex'].dtype != np.int64 and X['sex'].dtype != np.float64:
            column_contents = X['sex'].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            X['sex'] = list(map(convert_to_int, X['sex']))

            return X
data = handle_non_numerical_data(X)
print(X.head())	
from sklearn import svm
clf=svm.LinearSVC()
clf.fit(X,Y)
test = pd.read_csv('xtest.csv')
Z = test[columns_train]
Z['age'] = X['age'].fillna(X['age'].median())
X['age'].isnull().sum()
Z = handle_non_numerical_data(Z)
print(X.head())
pred = clf.predict(Z)
print pred
for i in pred:
    print(i)

np.savetxt('solution.csv',pred,delimiter=' ')