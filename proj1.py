import pandas as pd
import numpy as np
import os

os.chdir(("C:\\Users\\arjun\\OneDrive\\Documents\\Talking Totem\\Reports\\Datasets"))

FullRaw = pd.read_csv(("Techs_prediction.csv"))

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(FullRaw,test_size = 0.3, random_state = 123)

Train_X = Train.drop(['Technicians required/Daily'],axis =1)
Train_Y = Train['Technicians required/Daily']
Test_X = Test.drop(['Technicians required/Daily'],axis =1)
Test_Y = Test['Technicians required/Daily']

from statsmodels.api import OLS

M1_Model = OLS(Train_Y,Train_X).fit()

Test_Pred = round(M1_Model.predict(Test_X),0)

from sklearn.metrics import r2_score

r_2 = r2_score(Test_Y,Test_Pred)


#RMSE(Root Mean squared error)
RMSE = np.sqrt(np.mean(Test_Y - Test_Pred))

#MAPE(Mean Absolute percentage error)
MAPE = np.mean(np.abs((Test_Y - Test_Pred)/Test_Y))*100

import pickle

pickle.dump(M1_Model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

model.predict([[10,9,1,20,180]])