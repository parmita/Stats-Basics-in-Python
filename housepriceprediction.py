

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import load_boston
boston= load_boston() 
print ("type -datastructure " ,type(boston)) #boston is a disctionary
print ("Keys are " ,boston.keys()) # printing the keys
print ("Shape is " ,boston.data.shape) # printing the shape
print ("Features are " ,boston.feature_names) # printing the features
print ("description are " ,boston.DESCR) # printing the description

bos = pd.DataFrame(boston.data) #convert boston.data into a pandas data frame
print (bos.head()) #column names are just numbers,hence i would replace
bos.columns = boston.feature_names
print (bos.head()) #column names are present
print (boston.target[:5]) #boston.target contains the housing prices.
bos['Price'] =boston.target
print (bos.head())

'''In this section I am going to  fit a linear regression model 
and predict the Boston housing prices. I will use the least squares method 
as the way to estimate the coefficients.
Y = boston housing price(also called “target” data in Python)

and

X = all the other features (or independent variables)'''

from sklearn.linear_model import LinearRegression
X=bos.drop('Price',axis=1)
model = LinearRegression()

'''Important functions to keep in mind while fitting a linear regression model are:

lm.fit() -> fits a linear model

lm.predict() -> Predict Y using the linear model with estimated coefficients

lm.score() -> Returns the coefficient of determination (R^2). A measure of how well observed outcomes are replicated by the model, as the proportion of total variation of outcomes explained by the model. '''
model.fit(X, bos['Price'])

print ("estimated intercept coefficient are " ,model.intercept_) # printing the intercept
print ("estimated Number of coeficient " ,len(model.coef_)) # printing the number of coeff
print ("estimated Number of coeficient " ,model.coef_ )# printing the number of coeff

#I then construct a data frame that contains features and estimated coefficients.

print (pd.DataFrame(list(zip(X.columns,model.coef_)),columns=['features', 'estimatedcoefficient']))

#As you can see from the data frame that there is a high correlation between RM and prices. Lets plot a scatter plot between True housing prices and True RM.

plt.scatter(bos['RM'],bos['Price'])
plt.xlabel("average number od rooms per dwelling -RM")
plt.ylabel('House Price')
plt.title('relationship RM vs Price')
plt.show()  #positive correlation between RM and housing prices.

print (model.predict(X)[0:5]) #These are my predicted housing prices.

#Then I plot a scatter plot to compare true prices and the predicted prices.

plt.scatter(bos['Price'],model.predict(X))
plt.xlabel("price Y input")
plt.ylabel('Predicted price ')
plt.title('price vs predicted price')
plt.show()  

#Lets calculate the mean squared error.

msefull= np.mean((bos['Price'] - model.predict(X)**2))
print ("Mean square error " ,msefull)


#Training and validation data sets

X_train = X[:-50]
X_test = X[-50:]

Y_train = bos['Price'][0:-50]
Y_test = bos['Price'][-50:]

print ("Shape of X_train is " ,X_train.shape) # printing the shape
print ("Shape of X_test is " ,X_test.shape) # printing the shape
print ("Shape of Y_train is " ,Y_train.shape) # printing the shape
print ("Shape of Y_test is " ,Y_test.shape) # printing the shape

#You can create training and test data sets manually, but this is not the right way to do, because you may be training your model on less expensive houses and testing on expensive houses.
#Scikit learn provides a function called train_test_split to do this.

X_train,X_test,Y_train,Y_test =sklearn.cross_validation.train_test_split(X,bos['Price'],test_size=0.33)
print ("Shape of X_train is " ,X_train.shape) # printing the shape
print ("Shape of X_test is " ,X_test.shape) # printing the shape
print ("Shape of Y_train is " ,Y_train.shape) # printing the shape
print ("Shape of Y_test is " ,Y_test.shape) # printing the shape

lm=LinearRegression()
lm.fit(X_train,Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)

msefulltrain= np.mean((Y_train - lm.predict(X_train)**2))
msefulltest= np.mean((Y_test - lm.predict(X_test)**2))


print ("Mean square error trainset " ,msefulltrain)
print ("Mean square error test set " ,msefulltest)

plt.scatter(lm.predict(X_train),lm.predict(X_train)-Y_train,c='b')
plt.scatter(lm.predict(X_test),lm.predict(X_test)-Y_test,c='g')
plt.xlabel("residual plot -training(blue),test (Green)")
plt.ylabel('residuals')
plt.title('residual plot')
plt.show()



















