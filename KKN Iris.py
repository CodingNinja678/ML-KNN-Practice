#Loading DATASET
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


iris=datasets.load_iris()
features=iris.data
labels=iris.target
#print(features[0],labels[0])
#print(iris.DESCR) #IRIS DESCRIPTION
#Training the classifier
clf=KNeighborsClassifier()
clf.fit(features,labels)#Every classifier has a fit and predict feature
prediction=clf.predict([[1,1,1,1]])#sepal length sepal width petal length petal width
#print(prediction)
#print (iris.feature_names)
print (iris.target)
#print (iris.labels)
#print(iris.data)