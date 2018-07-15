import numpy as np
import pandas as pd
import csv
from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("kc_house_data.csv")
labels = data['price']
conv_dates = [1 if values == 2018 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)
x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)
gnb=GaussianNB()
y_predict=gnb.fit(x_train,y_train).predict(x_test)
acc=(y_test != y_predict).sum()
print("Number is %d"%(acc/len(y_test)))