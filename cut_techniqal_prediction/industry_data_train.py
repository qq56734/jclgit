import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error,roc_auc_score
from xgboost import plot_importance
from matplotlib import pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import numpy as np
import os

base_dir = os.path.abspath('.')

raw_data = pd.read_csv(os.path.join(base_dir, 'raw_data.csv'))
machine_data = pd.read_csv(os.path.join(base_dir, 'machine_data.csv'))




raw_data = raw_data.replace({'False': 0, 'True': 1, '单': 0, '双': 1 })
raw_data = pd.merge(raw_data, machine_data, on='machine_id', how='left')


rm_result_dict = {}
rm_accuracy_dict = {}
xgb_result_dict = {}
xgb_accuracy_dict = {}


goals = [
 'Material.CamParams.WorkSpeed',
'Material.CamParams.Cut.Height',
 'Material.CamParams.Cut.Focus',
'Material.CamParams.LiftHeight',
 'Material.CamParams.Cut.GasPressure',
'Material.CamParams.SlowLeadLength',
'Material.CamParams.SlowEndLength',
'Material.CamParams.SlowEndSpeed',
 'Material.CamParams.PierceFlags',
'Material.CamParams.SlowLeadSpeed',
'Material.CamParams.Cut.PwmRatio',
'Material.CamParams.PierceStepCount',
 'Material.CamParams.Cut.PwmFreq',
 'Material.CamParams.Cut.TimeStay',
#'Material.PwmRatioFunc.SmoothType',
#'Material.PwmRatioFunc.KnotCount',
'Material.CamParams.Cut.LaserCurrent',
#'Material.CamParams.PierceStyle',
#'Material.PwmFreqFunc.KnotCount',
'Material.CamParams.DelayBeforeLaserOff',
'Material.DefPwmFreqFunc',
'Material.DefPwmRatioFunc'
#'Material.PwmFreqFunc.SmoothType'
]


for i in [0, 2, 3, 4, 10, 12, -4]:





 predict_goal = goals[i]


 #the_x = ['fsmt', 'fsth', 'fsgs', 'fsns','fsnz', 'laser_model','power', 'Material.CamParams.Cut.GasType']
 the_x = ['fsmt', 'fsth','fsns','fsnz', 'laser_model','power', 'Material.CamParams.Cut.GasType']#+ ['Material.CamParams.Cut.Height', 'Material.CamParams.Cut.GasPressure', 'Material.CamParams.Cut.Focus']

 #the_x.remove(predict_goal)



 the_x.append(predict_goal)


 learn_data = raw_data[the_x]
 learn_droped = learn_data.dropna()





 cor = 1

 labelencoder_X = joblib.load(r'C:\Users\fscut\Desktop\ia\lables/' +  'fsmt' +  '.pkl')
 learn_droped.loc[:,'fsmt'] = labelencoder_X.transform(learn_droped.loc[:,'fsmt'])

 #labelencoder_y = joblib.load(r'C:\Users\fscut\Desktop\ia\lables/' +  'fsgs' +  '.pkl')
 #learn_droped.loc[:,'fsgs'] = labelencoder_y.transform(learn_droped.loc[:,'fsgs'])


 labelencoder_z = joblib.load(r'C:\Users\fscut\Desktop\ia\lables/' +  'laser_model' +  '.pkl')
 learn_droped.loc[:,'laser_model'] = labelencoder_z.transform(learn_droped.loc[:,'laser_model'])




 labelencoder_f = joblib.load(r'C:\Users\fscut\Desktop\ia\lables/' +  'Material.CamParams.Cut.GasType' +  '.pkl')
 learn_droped.loc[:,'Material.CamParams.Cut.GasType'] = labelencoder_f.transform(learn_droped.loc[:,'Material.CamParams.Cut.GasType'])






 if (cor == 0):
     labelencoder_e = LabelEncoder()
     learn_droped.loc[:,predict_goal] = labelencoder_e.fit_transform(learn_droped.loc[:,predict_goal])
     joblib.dump(labelencoder_e,r'C:\Users\fscut\Desktop\ia\lables/' +  predict_goal +  '.pkl')




 X_train, X_test, y_train, y_test = train_test_split(learn_droped.drop(predict_goal, 1), learn_droped[predict_goal], test_size = 0.01, random_state = 0)



 data_matrix = xgb.DMatrix(X_train,y_train)

 params = {'learning_rate': 0.1, 'objective': 'reg:squarederror',
                 'max_depth': 10, 'alpha': 1}



 cv_results = xgb.cv(dtrain=data_matrix, params=params, nfold=5,
                     num_boost_round=500,early_stopping_rounds=8,metrics="rmse", as_pandas=True, seed=123)

 print('start training')

 xg_reg= xgb.XGBRegressor(
     objective='reg:squarederror',
     learning_rate=0.1,
     max_depth=8,
     n_estimators=cv_results.shape[0],
     alpha =1
 )

 xg_reg.fit(X_train,y_train)



 y_predict = xg_reg.predict(X_test)
 #y_predict = np.round(y_predict, decimals=1)
 print(mean_squared_error(y_predict,y_test))







 joblib.dump(xg_reg,os.path.join(base_dir, 'models',  'xgb',predict_goal +  '.pkl'))













