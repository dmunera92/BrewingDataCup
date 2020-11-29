# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:15:40 2020

@author: usuario
"""

import pandas as pd
import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
pd.set_option('display.max_columns', None)
from aux_funct import *


if __name__ == "__main__":
    
    print("Leyendo Archivos")
    # Lectura y Transformación de datos
    df_total,test_data = data_reader()
    # Transformación de datos
    pivot_data,df_marcas = data_process_1(df_total)
    pivot_cuant = pivot_cuant(df_marcas)
    pivot_fecha_final = pivot_fecha(df_total)
    testdf5 = test_data_f5(test_data,pivot_data,pivot_fecha_final)
    # Datos de Entrenamiento y Output Modelo
    X,y = train_data(pivot_cuant,pivot_data)
    ## Predicción del Modelo.
    y_1 = model_predict(X,y)
    # Salida Final del Modelo 
    output_model = final_df(test_data,y_1,pivot_fecha_final,testdf5)
    
    