# Main Aim : Predict Employee will be fired out from our company or not 

import pandas as pd  # to read the dataset
import numpy as np   # ndim array
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv("EMPdataset.csv")
df.head()  # display first 5  , tail - for last 5 
df.shape
df.columns  # what columns available in dataset
df.info()   # to check which columns are categorical

#checking null value or not
df.isnull().sum()   #summ all data and give value 0 - no null value

df.dropna()   # to remove the null value
df.describe() # view some statistics

sns.countplot(x= 'Attrition', data= df)

df['Attrition'].value_counts()

# drop some unwanted columns
df.columns
df.drop('EmployeeNumber', axis=1, inplace=True)   # permanently affect the dataframe
df.drop('EmployeeCount', axis=1, inplace=True)
df.drop('Over18', axis=1, inplace=True)
df.drop('StandardHours', axis=1, inplace=True)

df.shape

# convert categorical value in 0 and 1
categorical_column = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender',
			'JobRole', 'MaritalStatus', 'OverTime']

dataen = df.copy(deep=True)  # deep includes integer values also

# Pre Processing - convert categorical into 0 and 1
from sklearn.preprocessing import LabelEncoder #object = 0,1,2
laben = LabelEncoder()

for col in categorical_column:
	dataen[col] = laben.fit_transform(df[col])

dataen.head()
dataen.info()

# divide data into train and test
x = dataen.iloc[:, dataen.columns != 'Attrition'].values
y = dataen.iloc[:, dataen.columns == 'Attrition'].values

print(x)

# Train and Test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# import Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10, random_state=10)
rf.fit(xtrain, ytrain)
rfpredict = rf.predict(xtest)   #ytest original output
print(rfpredict)

from sklearn import metrics
rfacc = metrics.accuracy_score(ytest, rfpredict)
print(rfacc)

#confusion matrix
cnf = metrics.confusion_matrix(ytest,  rfpredict)
print(cnf)

label = ['no', 'yes']
sns.heatmap(cnf, annot=True, cmap='YlGnBu', fmt='.3f', xticklabels=label, yticklabels=label)
