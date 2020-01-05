#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn import model_selection,tree, metrics
from sklearn.externals.six import StringIO 
import pydotplus
import matplotlib.pyplot as plt
from IPython.display import Image


# In[12]:


data = pd.read_excel(r'/Users/moshadab/Documents/BITS/DM/Assignments/Assignment-1/C2DM6/input/bank_data_for_cs.xlsx')
print('Dataset shape: ', data.shape)

 
# copy dataframe for future use
dataset = data.copy()

 
# gives null values
nullValues = dataset[dataset.isnull().any(axis=1)]


# get the unique values from dataset to check if any redundant values present
for col in dataset:
    uniqueData = dataset[col].unique()
    print('Column name: ', col)
    print(uniqueData)

    
# check if a column contains particular string and
# if true gives the index of that string    
mask = dataset['housing loan?'].str.contains(r'xxxyy', na=True)
print(mask.sum())
if mask.sum():
    redundantIndex = dataset.loc[dataset['housing loan?'] == 'xxxyy'].index.values

   
# replace the redundant data to pandas NaN
Newdata = dataset.replace(dataset['housing loan?'].values[redundantIndex],np.NaN)

 
# get the number of null values present in each column
null = data.isnull().sum()
#print(null.sum())

 
# change null value to the most frequent occuring data
if null.sum()!=0:
    dataNew = Newdata.apply(lambda x: x.fillna(x.value_counts().index[0]))

 
dataNew['y'],class_names = pd.factorize(data['y'])
dataNew['job'],_ = pd.factorize(data['job'])
dataNew['marital status '],_ = pd.factorize(data['marital status '])
dataNew['education'],_ = pd.factorize(data['education'])
dataNew['credit default?'],_ = pd.factorize(data['credit default?'])
dataNew['housing loan?'],_ = pd.factorize(data['housing loan?'])
dataNew['Personal loan'],_ = pd.factorize(data['Personal loan'])
#print(dataNew.head(3))

 
# select dependent and independent features
X = dataNew.iloc[:, :-1] # independent vectors
Y = dataNew.iloc[:, -1]   # dependent/ target vector

 
# split data randomly into 70% training and 30% test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=0)

 
# train the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
dtree.fit(X_train, y_train)

 
# use the model to make predictions with the test data
y_pred = dtree.predict(X_test)

 
# model performance
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))

 
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


precision =  metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_Score = metrics.f1_score(y_test, y_pred)

 


# confusion matrix
confuseMetrics = metrics.confusion_matrix(y_test, y_pred)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confuseMetrics)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
#ax.set_xticklabels([''] + labels)
#ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

 

# graph visualization
feature_names = X.columns

dot_data = StringIO()
 
tree.export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True,
                                special_characters=True,
                                feature_names=feature_names, 
                                class_names=class_names)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(r'/Users/moshadab/Documents/BITS/DM/Assignments/Assignment-1/Assignment_CHN/output/tree.png')
Image(graph.create_png())


# In[ ]:




