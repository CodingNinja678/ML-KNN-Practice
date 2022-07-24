import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap=ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris=datasets.load_iris()
X,y=iris.data,iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
#print(X_train.shape)#120 is the no of samples 4i s no of features for each sample
#print(X_train[0])
#print(y_train.shape)
#print(y_train)

plt.figure()
plt.scatter(X[:,2],X[:,3],c=y,cmap=cmap,edgecolor='k',s=20)
plt.show()
a=[1,1,2,2,3,4,5,6]
from collections import Counter
most_common=Counter(a).most_common(1)
print(most_common)