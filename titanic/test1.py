# import pandas and numpy
import pandas as pd
from pandas import Series,DataFrame
import numpy as np

# Loading data and printing first few rows
train = pd.read_csv('xtrain.csv')
test = pd.read_csv('xtest.csv')
# Previewing the statistics of training data and test data
test.head(2)
# print the names of the columns in the data frame
print("In Training Data missing, columns with missing values:")
# Retain columns that are of interest and discard the rest (such as Id, Name, Cabin, and Ticket number)
newcols = ['survived', 'pclass', 'sex', 'age', 'sibsp','parch', 'fare','embarked']
train = train[newcols]
# Check which columns have missing data
for column in train.columns:
    if np.any(pd.isnull(train[column])) == True:
        print(column) 
# print the names of the columns in the data frame
print("In test Data missing, columns with missing values:")
# Retain columns that are of interest and discard the rest (\Name, Cabin and Ticket number)
newcols = ['pclass', 'sex', 'age', 'sibsp','parch', 'fare', 'embarked']
test = test[newcols]
# Check which columns have missing data
for column in test.columns:
    if np.any(pd.isnull(test[column])) == True:
        print(column)

# Filling missing age data with median values
train["age"] = train["age"].fillna(train["age"].median())

# data cleaning for Embarked
print (train["embarked"].unique())
print (train.embarked.value_counts())
train["embarked"] = train["embarked"].fillna('S')

# Filling missing age data with median values of trainging set
test["age"] = test["age"].fillna(train["age"].median())

# filling fare data with median of training set
test["fare"] = test["fare"].fillna(train["fare"].median())

for df in [train, test]:
    df['FamilySize'] = df['parch'] + df['sibsp'] + 1

def filter_family_size(x):
    if x == 1:
        return 'Solo'
    elif x < 4:
        return 'Small'
    else:
        return 'Big'

for df in [train, test]:
    df['FamilySize'] = df['FamilySize'].apply(filter_family_size)
# matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
%matplotlib inline
plt.rcParams.update({'font.size': 22})

# Check with Pclass and Embarked
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))
ax1.set_title('Survival rates for Passenger Classes')
sns.barplot(x='pclass', y='survived', hue='sex', data=train, ax=ax1)

ax2.set_title('Survival rates for Port Embarked')
sns.barplot(x='embarked', y='survived', hue='sex', data=train, ax=ax2)
sns.despine()
sns.set(font_scale=1.4)
ax2.legend_.remove()
ax1.legend(loc='upper right')
plt.show()

# Check with Age
g = sns.FacetGrid(train, col="survived", hue='sex', size=5, aspect = 1.2)
g.map(sns.kdeplot, "age", shade=True).add_legend().fig.subplots_adjust(wspace=.3)
sns.despine()
sns.set(font_scale=2)
plt.show()


# Check with Fare
g = sns.FacetGrid(train, col="survived", hue='sex', size=5, aspect = 1.2)
g.map(sns.kdeplot, "fare", shade=True).add_legend().fig.subplots_adjust(wspace=.3)
sns.set(font_scale=2)
sns.despine()
plt.show()


# Family Size
sns.barplot(x='FamilySize', y='survived' , data=train, order = ['Solo', 'Small', 'Big'])
sns.set(font_scale=1.5)
plt.show()

train.loc[train["embarked"] == 'S', "embarked"] = 0
train.loc[train["embarked"] == 'C', "embarked"] = 1
train.loc[train["embarked"] == 'Q', "embarked"] = 2

test.loc[test["embarked"] == 'S', "embarked"] = 0
test.loc[test["embarked"] == 'C', "embarked"] = 1
test.loc[test["embarked"] == 'Q', "embarked"] = 2

# convert female/male to numeric values (male=0, female=1)
train.loc[train["sex"]==True,"sex"]=0
train.loc[train["sex"]==False,"sex"]=1

test.loc[test["sex"]==True,"sex"]=0
test.loc[test["sex"]==False,"sex"]=1

# convert family size to numeric values

train.loc[train["FamilySize"] == 'Solo', "FamilySize"] = 0
train.loc[train["FamilySize"] == 'Small', "FamilySize"] = 1
train.loc[train["FamilySize"] == 'Big', "FamilySize"] = 2

test.loc[test["FamilySize"] == 'Solo', "FamilySize"] = 0
test.loc[test["FamilySize"] == 'Small', "FamilySize"] = 1
test.loc[test["FamilySize"] == 'Big', "FamilySize"] = 2

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

# columns we'll use to predict outcome
features = ['pclass', 'age', 'FamilySize', 'fare', 'embarked']
label = 'sex'

# instantiate the model
logreg = LogisticRegression()

# perform cross-validation
#print(cross_val_score(logreg, train[features], train[label], cv=10, scoring='accuracy').mean())

# Apply our prediction to test data
logreg.fit(train[features], train[label])
prediction = logreg.predict(test[features])

# Create a new dataframe with only the columns Kaggle wants from the dataset
submission_DF = pd.DataFrame({ 
    "PassengerId" : test["PassengerId"],
    "Survived" : prediction
    })
print(submission_DF.head(2))