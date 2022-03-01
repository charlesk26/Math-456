import pandas
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split

df = pandas.read_csv("stature_hand_foot.csv")
X = df[['hand', 'foot']]
Y = df['height']
# with sklearn
# regr = linear_model.LinearRegression()
# regr.fit(X, Y)

clf = linear_model.LinearRegression()
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3)
clf.fit(x_train, y_train)
clf.predict(x_test)
print(clf.score(x_test,y_test))
print('Intercept: \n', clf.intercept_)
print('Coefficients: \n', clf.coef_)
