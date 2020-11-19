#!/usr/bin/env python
# coding: utf-8

import os
import joblib
import numpy as np
import pandas as pd

class data_for_input:

    def __init__(self, fsmt, fsth, fsns, fsnz, laser_model, power, GasType):
        self.fsmt = fsmt
        self.fsth = fsth
        self.fsns = fsns
        self.fsnz = fsnz
        self.laser_model = laser_model
        self.power = power
        self.GasType = GasType



# input array column names: fsmt, fsth, fsgs, fsns, fsnz, laser_model, power, Material.CamParams.Cut.GasType, Material.OEMCode
# output array column names: ['Material.CamParams.WorkSpeed', 'Material.CamParams.SlowLeadLength'] + rfr_objects + rfc_objects


def get_prediction(data_input):


    x_valid = pd.DataFrame(data=[
        [data_input.fsmt, data_input.fsth,  data_input.fsns, data_input.fsnz, data_input.laser_model,
         data_input.power, data_input.GasType]],
        columns=['fsmt', 'fsth',  'fsns', 'fsnz', 'laser_model', 'power',
                 'Material.CamParams.Cut.GasType'])

    x_valid.loc[:, 'fsmt'] = labelencoder_fsmt.transform(x_valid.loc[:, 'fsmt'])

    x_valid.loc[:, 'laser_model'] = labelencoder_model.transform(x_valid.loc[:, 'laser_model'])


    x_valid.loc[:, 'Material.CamParams.Cut.GasType'] = labelencoder_GasType.transform(
        x_valid.loc[:, 'Material.CamParams.Cut.GasType'])

    y_predict = xgb_Workspeed.predict(x_valid)



    for i in range(6):
        predict_value = xgb_models[i].predict(x_valid)

        y_predict = np.vstack((y_predict, predict_value))

    # write to xml


    return y_predict.T

base_dir = os.path.abspath('.')


# load XGB models


xgb_Workspeed = joblib.load(os.path.join(base_dir, 'models', 'xgb', 'Material.CamParams.WorkSpeed.pkl'))


reamin_objects = ['Material.CamParams.Cut.Focus', 'Material.CamParams.LiftHeight', 'Material.CamParams.Cut.GasPressure', \
                  'Material.CamParams.Cut.PwmRatio',  'Material.CamParams.Cut.PwmFreq', 'Material.CamParams.Cut.LaserCurrent'
               ]



xgb_models = []




for object in reamin_objects:
    xgb_models.append(joblib.load(os.path.join(base_dir, 'models', 'xgb', object + '.pkl')))


# laod lablers


os.path.join(base_dir, 'labels', 'fsmt.pkl')

labelencoder_fsmt = joblib.load(os.path.join(base_dir, 'labels', 'fsmt.pkl'))


labelencoder_model = joblib.load(os.path.join(base_dir, 'labels', 'laser_model.pkl'))


labelencoder_GasType = joblib.load(os.path.join(base_dir, 'labels', 'Material.CamParams.Cut.GasType.pkl'))





# fsmt, fsth, fsns, fsnz, laser_model, power, GasType

gastype_dict = {
    '空气':1, '氧气':2, '氮气':3, '高压空气':4 , '高压氧气':5,  '高压氮气':6, '低压气':81, '高压气': 82, '默认':0
}
fsnz_dict = {'单':0, '双': 1}


data_input = data_for_input('碳钢', 2.0, 1.2, fsnz_dict['双'], 'IPG', 1500, gastype_dict['氧气'])


# Material.CamParams.WorkSpeed 'Material.CamParams.Cut.Focus', 'Material.CamParams.LiftHeight', 'Material.CamParams.Cut.GasPressure', \
# 'Material.CamParams.Cut.PwmRatio', 'Material.CamParams.Cut.PwmFreq', 'Material.CamParams.Cut.LaserCurrent'
y = get_prediction(data_input)

print(y)


