import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# load the data
df = pd.read_csv('mushrooms.csv')

# There are no null columns in the dataset
df[df.isnull().any(axis=1)]

from sklearn.model_selection import train_test_split


# split the data int x(training data) and y (results)
y = df['class']
x = df.drop(['class'], axis=1)
x = pd.get_dummies(x) #Convert categorical variable into dummy/indicator variables
y = pd.get_dummies(y)  #Convert categorical variable into dummy/indicator variables
x.info()
y.info()
# x.info()
# y.info()
# x.dtypes

from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

parameters = {'criterion':('gini', 'entropy'), 
              'min_samples_split':[2,3,4,5], 
              'max_depth':[9,10,11,12],
              'class_weight':('balanced', None),
              'presort':(False,True),
             }


tr = tree.DecisionTreeClassifier()
gsearch = GridSearchCV(tr, parameters)
gsearch.fit(X_train, y_train)
model = gsearch.best_estimator_
print (model)

score = model.score(X_test, y_test)
score


import graphviz
dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=X_test.columns,
                               class_names=y_test.columns,
                               filled=True, rounded=True,
                              special_characters=True)
graph = graphviz.Source(dot_data)
print (graph)
print(dot_data) 