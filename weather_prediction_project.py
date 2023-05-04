
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

get_ipython().run_line_magic('matplotlib', 'inline')



df = pd.read_csv("C:/Users/chanu/Desktop/bkp/DATASET/weather_prediction_project.csv")
df.head(10)


#printing last 10 rows
df.tail(10) 


#dimensions of the data
df.shape 


#print the columns/features of the data
df.columns



# listing the numericale features
num_cols=df.select_dtypes(include=np.number).columns
num_cols


# listing the categorical features
cat_cols=df.select_dtypes(include='object').columns
cat_cols



#basic info of the dataset
df.describe() 

df.isnull().sum()


data=df.drop(['date_time', 'moonrise', 'moonset', 'sunrise', 'sunset'],axis='columns')
data



def cap_data(data):
    for col in data.columns:
        print("\n\n capping the \n",col)
        if (((data[col].dtype)=='float64') | ((data[col].dtype)=='int64')):
            
            q1=data[col].quantile(0.25)
            q3=data[col].quantile(0.75)
            iqr=q3-q1
            lower,upper=(q1-(iqr*1.5)),(q3+(iqr*1.5))    
            print("q1=",q1,"q3=",q3,"iqr=",iqr,"lower=",lower,"upper=",upper) 
            data[col][data[col] <= lower] = lower
            data[col][data[col] >= upper] = upper
            print("\n",data[col][data[col] <= lower] )
            print("\n",data[col][data[col] >= upper] )
            
        else:
            data[col]=data[col]
    return data

final_df=cap_data(data)


x=data.drop(['tempC','totalSnow_cm','uvIndex','uvIndex.1','moon_illumination','DewPointC','FeelsLikeC','WindChillC','WindGustKmph','visibility','winddirDegree'],axis='columns')
x



y=data['tempC']
y


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)



x_train.head()



x_test.head()



model=LinearRegression()
model.fit(x_train,y_train)
prediction = model.predict(x_test)



# calculating error
np.mean(np.absolute(prediction-y_test))



print('Variance score: %.2f' % model.score(x_test,y_test))


for i in range(len(prediction)):
  prediction[i]=round(prediction[i],2)
pd.DataFrame({'Actual':y_test,'Prediction':prediction,'diff':(y_test-prediction)})


print("Mean absolute error: %.2f" % np.mean(np.absolute(prediction - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((prediction - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_test,prediction ) )


from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
model.fit(x_train,y_train)
prediction = model.predict(x_test)


# calculating error
np.mean(np.absolute(prediction-y_test))


print('Variance score: %.2f' % model.score(x_test,y_test))


for i in range(len(prediction)):
  prediction[i]=round(prediction[i],2)
pd.DataFrame({'Actual':y_test,'Prediction':prediction,'diff':(y_test-prediction)})


print("Mean absolute error: %.2f" % np.mean(np.absolute(prediction - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((prediction - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_test,prediction ) )


from sklearn.ensemble import RandomForestRegressor
regr=RandomForestRegressor(max_depth=90,random_state=0,n_estimators=100)
regr.fit(x_train,y_train)


prediction3=regr.predict(x_test)
np.mean(np.absolute(prediction3-y_test))


regr.predict([[15,11,7.1,12.0,27.0,64,0.0,1020,10]])

# model.score(x_test,y_test)*100
print('Variance score: %.2f' % regr.score(x_test,y_test))

for i in range(len(prediction3)):
  prediction3[i]=round(prediction3[i],2)
pd.DataFrame({'Actual':y_test,'Prediction':prediction3,'diff':(y_test-prediction3)})

from sklearn.metrics import r2_score
print("Mean absolute error: %.2f" % np.mean(np.absolute(prediction3 - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((prediction3 - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_test,prediction3 ) )


import pickle


with open('weather_prediction_pickle','wb') as f:
    pickle.dump(regr,f)


with open('weather_prediction_pickle','rb') as f:
    model_predict=pickle.load(f)


model_predict.predict([[15,11,7.1,12.0,27.0,64,0.0,1020,10]])




