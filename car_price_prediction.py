import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
from sklearn import metrics

cars = pd.read_csv('car.csv')
# print(cars.head())
# print(cars.shape)
#print(cars.info())
#print(cars.isnull().sum())

cars.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
cars.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
cars.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

# print(cars)

X = cars.drop(['Car_Name','Selling_Price'],axis=1)
Y = cars['Selling_Price']

# print(X)
# print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,random_state=2)

linear_model = LinearRegression()

linear_model.fit(X_train,Y_train)

training_predit = linear_model.predict(X_train)

error_score = metrics.r2_score(Y_train,training_predit)

# coef of determination that tells how accurate the explanation of variance is and how accurate the training model is)")
print("R Square value: " + str(error_score))

plt.scatter(Y_train,training_predit)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()