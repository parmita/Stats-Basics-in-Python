import numpy as np # fundamental package for scientific computing with Python. 
import pandas as pd #for data manipulation and analysis

data = pd.read_csv('Student_Vs_Marks.csv') # Specify the path of file from your local drive

mean = np.mean(data['Marks']) # finding the mean of the marks
st_dev = np.std(data['Marks']) # finding the standard deviation of the marks

print ("Mean is  %s." % mean)
print ("standard deviation is  %s." % st_dev)

median = np.median(data['Marks'])
maximum = np.max(data['Marks'])
minimum = np.min(data['Marks'])

print ("median is  %s." % median)
print ("maximum is  %s." % maximum)
print ("minimum is  %s." % minimum)

print (data.describe()) # gives the Summary statistics
print (data.shape) # dimension of the data

#Visualize the Statistics
import matplotlib.pyplot as plt #is a plotting library

# a simple  bar  plot
data.plot(kind='bar',x='Student',y='Marks')
plt.show()

# a simple line  plot
data.plot(kind='line',x='Student',y='Marks')
plt.show()

# basic Boxplot
plt.boxplot(data['Marks'])
plt.show()



# Lets start with Matrix
# initializing matrices 
x = np.array([[1, 2], [4, 5]]) 
y = np.array([[7, 8], [9, 10]]) 

# using add() to add matrices 
print ("The element wise addition of matrix is : ") 
print (np.add(x,y)) 

# using subtract() to subtract matrices 
print ("The element wise subtraction of matrix is : ") 
print (np.subtract(x,y))

# using divide() to divide matrices 
print ("The element wise division of matrix is : ") 
print (np.divide(x,y))


#  code to demonstrate matrix operations 
# multiply() and dot() 

# using multiply() to multiply matrices element wise 
print ("The element wise multiplication of matrix is : ") 
print (np.multiply(x,y))

# using dot() to multiply matrices 
print ("The product of matrices is : ") 
print (np.dot(x,y))

# using sqrt() to print the square root of matrix 
print ("The element wise square root is : ") 
print (np.sqrt(x))

# using sum() to print summation of all elements of matrix 
print ("The summation of all matrix element is : ") 
print (np.sum(y))

# using sum(axis=0) to print summation of all columns of matrix 
print ("The column wise summation of all matrix  is : ") 
print (np.sum(y,axis=0))