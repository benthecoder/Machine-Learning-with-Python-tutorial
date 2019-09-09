import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

print(data.head())

# defining the attribute
predict = "G3"

x = np.array(data.drop([predict], 1))  # features
y = np.array(data[predict])  # labels

y_test: object
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)  # 10% test size

# x_train is the x section and y_train is the y section
# x_test and y_test are testing the accuracy of the algorithm

# LINEAR REGRESSION
linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)  # fitting algorithm onto test set (training)
acc = linear.score(x_test, y_test)  # scoring the model
print(acc)  # accuracy above 80% is good

# making prediction

print('Coefficients: ', linear.coef_)  # slope of values
print('Intercepts: ', linear.intercept_)  # intercept

score_prediction = linear.predict(x_test) # list of all predictions
for x in range(len(score_prediction)):
    print(int(score_prediction[x]), x_test[x], y_test[x])
