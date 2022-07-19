#from urllib import request
#url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'
#request.urlretrieve(url,'tele-cust.csv')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing



df=pd.read_csv('tele-cust.csv')
df.columns

df.head()

df['custcat'].value_counts()
df.hist()
#plt.show()

df.hist(column='income')
plt.savefig('income.png',dpi=300)
#plt.show()


#converting the data to numpy arrays

x=df[['region', 'tenure', 'age', 'marital', 'address', 'income','ed','employ', 'retire', 'gender', 'reside']].values
x[:5]
y=df['custcat'].values
y[:5]

#normalising data: mean is 0 and variance is 1

x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

#x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))
x[0:5]

#test/train split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=4)

from sklearn.neighbors import KNeighborsClassifier
k=7
neigh=KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
neigh
y_pred=neigh.predict(x_test)
y_pred[0:2]


#accuracy score
from sklearn import metrics
print(metrics.accuracy_score(y_train,neigh.predict(x_train)))
print(metrics.accuracy_score(y_test,y_pred))




