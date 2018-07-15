import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import ensemble


main_data=pd.read_csv("kc_house_data.csv")
converting_dates = [1 if values == 2014 else 0 for values in main_data.date]
main_data['date']= converting_dates
x_set=main_data.drop(['id','price'],axis=1)
y_set=main_data['price']
x_trainingSet,x_testingSet,y_trainingSet,y_testingSet = train_test_split(x_set,y_set,test_size=0.15,random_state=2)


Log_regression=LogisticRegression()
Log_regression.fit(x_trainingSet, y_trainingSet)
result_reg=Log_regression.score(x_testingSet, y_testingSet)
print("Accuracy of Logistic Regression Model in percentage : ",result_reg*100)

classification= ensemble.GradientBoostingRegressor(n_estimators=400,max_depth=5,min_samples_split = 2,learning_rate=0.7,loss='ls')

classification.fit(x_trainingSet,y_trainingSet)
result_GBreg=classification.score(x_testingSet,y_testingSet)
print("Accuracy of Logistic Regression with Gradient Booster in percentage : ",result_GBreg*100)
