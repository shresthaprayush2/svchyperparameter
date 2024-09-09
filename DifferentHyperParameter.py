#Hyperparameters are the tunable parameters that help tune and customize the Scikit learn model
#Different paramter have different hyperparameters
#In this tutorial we will explore the hyperparameters of SVC algorithm
#SVC algorithm is the classification alogirthm

#Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

#Creating a function to plot the data
def plotSVC(title,svc):
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  h = (x_max / x_min)/100
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
  plt.subplot(1, 1, 1)
  Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
  plt.xlabel('Sepal length')
  plt.ylabel('Sepal width')
  plt.xlim(xx.min(), xx.max())
  plt.title(title)
  plt.savefig(f'HyperParameterTuningOfSVC/{title}.png')


#Using multiple kernel
#Kernels are used describe the plane. Linear kerner gives a linear plane which is a line in 2D
#rbf and poly gives non linear hyperplane
#Using rbf and poly increases the complexity as compared to linear
kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
  svc = SVC(kernel=kernel).fit(X, y)
  plotSVC('kernel=' + str(kernel),svc)


#gamm is the parameter used for nolinear hyperplanes
# the higher the gamma the more it tries to fit the dataset exactly
#higer gamma can cause overfitting

gammas = [0.1, 1, 10, 100]
for gamma in gammas:
   svc = SVC(kernel='rbf', gamma=gamma).fit(X, y)
   plotSVC('gamma=' + str(gamma),svc)


#C is the parameter for error term, It controls the tradeoff between smooth decision boundry and classifyin
#Training points correclty
#Increasing the C can also cause overfitting
cset = [0.1, 1, 10, 100, 1000]
for c in cset:
   svc = SVC(kernel='rbf', C=c).fit(X, y)
   plotSVC('C=' + str(c),svc)


#Another parameter is degree
#This is used in polynomial kernel
#It is basically the degree of the hyper plane
# the higher the degree the complex the algorithm

degrees = [0, 1, 2, 3, 4, 5, 6]
for degree in degrees:
   svc = SVC(kernel="poly", degree=degree).fit(X, y)
   plotSVC("degree=" + str(degree),svc)


#Finding the best hypermeter, to find the best hypermeter use use Grid Search
# Grid search traverses through all the hyper paramters combination and finds the best
# Note that this will take time depending upon your machine specification

degreeG = [0, 1, 6]
cSetG = [0.1, 1, 10]
gammaG = [0.1, 1, 10]
kernelsG = ['linear', 'rbf', 'poly']



from sklearn.model_selection import GridSearchCV
model = SVC()
gridSearch = GridSearchCV(estimator=model,param_grid={'degree':degreeG,'kernel':kernelsG,'C':cSetG,'gamma':gammaG},cv=4)
gridSearch.fit(X,y)
print(f'The best parameter is {gridSearch.best_params_}')
print(f'The best score is {gridSearch.best_score_}')
print(f'The best estimator is {gridSearch.best_estimator_}')



