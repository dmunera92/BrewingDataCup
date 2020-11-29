# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:19:52 2020

@author: usuario
"""

## funciones auxiliares



def clasificar_marca(x):
    
    if x["Marca2"] == "Marca_20" and x["Cupo2"] == "Cupo_3" and x["CapacidadEnvase2"] == "CapacidadEnvase_9":
        return "Marca1"
    elif x["Marca2"] == "Marca_16" and x["Cupo2"] == "Cupo_2" and x["CapacidadEnvase2"] == "CapacidadEnvase_10":
        return "Marca2"
    elif x["Marca2"] == "Marca_9" and x["Cupo2"] == "Cupo_3" and x["CapacidadEnvase2"] == "CapacidadEnvase_12":
        return "Marca3"
    
    elif x["Marca2"] == "Marca_38" and x["Cupo2"] == "Cupo_2" and x["CapacidadEnvase2"] == "CapacidadEnvase_10":
        return "Marca_Inno1" 
        
    elif x["Marca2"] == "Marca_39" and x["Cupo2"] == "Cupo_2" and x["CapacidadEnvase2"] == "CapacidadEnvase_10":
        return "Marca_Inno2"
    else:
        return "MarcaDif"


def data_reader():
    
    current_directory = os.getcwd()
    dir_data = os.path.join(current_directory,"data")
    files_in_dir = os.listdir(dir_data)

    df = pd.read_csv(files_in_dir[0],sep = ";")
    df_ventas = pd.read_csv(files_in_dir[1],sep = ";")
    test_data = pd.read_csv(files_in_dir[2],sep = ";")
    
    df_total = df_ventas.merge(df,how = "inner",on = "Cliente")
    df_total["nm"] = df_total.apply(lambda x: clasificar_marca(x),axis = 1)


    return  df_total,test_data
    
    
    
def data_process_1(df):
    
    tiempo_cliente = []
    clientes = []
    for cliente in df.Cliente.unique():
        clientes.append(cliente)
        tiempo_cliente.append(len(df[df.Cliente==cliente][['Año','Mes','Cliente']].drop_duplicates()))

    tiempo_clientes = pd.DataFrame({'Cliente':clientes,'Meses':tiempo_cliente})

    df_marcas = df[df.nm != "MarcaDif"].copy()
    
    c3 = df_marcas.groupby(['Cliente','nm']).size().to_frame('meses_pedido').reset_index()
    union = c3.merge(tiempo_clientes, how = 'left', on ='Cliente')
    union['prob'] = union.meses_pedido/union.Meses
    
    pivot_data = union.pivot(index = 'Cliente', columns='nm',values='prob').fillna(0).reset_index()
    
    return pivot_data,df_marcas

def pivot_cuant(df):
    vv = ['Cliente','nm','Volumen','disc','nr']
    media_cuantitativas = df[vv].groupby(['Cliente','nm']).agg({'Volumen':'mean','disc':'mean','nr':'mean'}).reset_index()
    pivot_cuatitativas = pd.get_dummies(media_cuantitativas , columns=['nm'])
    pivot_cuatitativas = pivot_cuatitativas.loc[~(pivot_cuatitativas.Volumen==0)]   

    pivot_cuatitativas.columns=['Cliente', 'Volumen', 'disc', 'nr', 'Marca1', 'Marca2',
       'Marca3', 'Marca_Inno1', 'Marca_Inno2']

    pivot_cuatitativas= pivot_cuatitativas.groupby('Cliente').agg({'Volumen':'mean',
                                           'disc':'mean','nr':'mean','Marca1':'sum',
                                           'Marca2':'sum','Marca3':'sum',
                                           'Marca_Inno1':'sum','Marca_Inno2':'sum'}).reset_index()
    
    return pivot_cuatitativas

def pivot_fecha(df):
    
    max_fecha_cliente=df.groupby('Cliente')['fecha'].max().reset_index()
    max_fecha_marca = df.groupby(['Cliente','nm'])['fecha'].max().reset_index()
    
    pivot_fecha = max_fecha_marca.pivot(index = 'Cliente', columns='nm',values='fecha').reset_index()
    pivot_fecha = pivot_fecha.merge(max_fecha_cliente, how = 'left', on='Cliente')
    columna_lista=[col for col in pivot_fecha.columns if col.startswith("Marca")]
    
    for columna in columna_lista:
        pivot_fecha[columna+'_r'] = pivot_fecha.fecha-pivot_fecha[columna]
        pivot_fecha[columna+'_r'] = pivot_fecha[columna+'_r'].dt.days.astype(float)/30
        conditions =[
        (pivot_fecha[columna+'_r']==0),
        (pivot_fecha[columna+'_r'].isna()),
        (pivot_fecha[columna+'_r']>0)
        ]
        valores = [1,0,1/pivot_fecha[columna+'_r']]
        pivot_fecha[columna+'_r'] = np.select(conditions, valores)
        
    columnas_fecha = [col for col in pivot_fecha.columns if col.endswith("_r") or col=='Cliente']
    pivot_fecha_final = pivot_fecha[columnas_fecha].copy()
    return pivot_fecha_final

def test_data_f5(test_data,pivot_data,pivot_fecha_final):
    
    test_data1 = pd.DataFrame(test_data['Cliente']).merge(pivot_data, how= 'left',on='Cliente')
    test_data1 = test_data1.fillna(test_data1.mean())
    test_data1.columns = test_data.columns
    
    test_data5_1 = test_data1.merge(pivot_fecha_final, how = 'left', on ='Cliente')
    
    return test_data5_1

def train_data(df1,df2):
    
    df1_select = df1[['Cliente','Volumen','disc','nr']]
    df2_select = df2.copy()
    df2_select.columns = ['Cliente','m1_f','m2_f','m3_f','m4_f','m5_f']
    
    X = df1_select.merge(df1_select,df2_select,how='left',on='Cliente')
    y= df1[['Marca1','Marca2','Marca3','Marca_Inno1','Marca_Inno2']]
    
    return X,y

def model_predict(X,y):
    
    regr = DecisionTreeRegressor(max_depth=30)
    regr.fit(X, y)
    y_1 = regr.predict(X)
    y_1.columns = ['Marca1','Marca2','Marca3','Marca_Inno1','Marca_Inno2']
    y_1['Cliente'] = X['Cliente']
    y_1 = y_1[['Cliente','Marca1','Marca2','Marca3','Marca_Inno1','Marca_Inno2']]
    
    return y_1
    
def final_df(df,y,pivot_fecha,testdf5):
    
    test_data6 = pd.DataFrame(df['Cliente']).merge(y,how='left', on='Cliente').fillna(0.5).merge(pivot_fecha,how='left', on='Cliente')
    entregables=['Marca1', 'Marca2', 'Marca3', 'Marca_Inno1', 'Marca_Inno2']
    
    for col_mar in entregables:
       test_data6[col_mar]= test_data6[col_mar]*testdf5[col_mar]#*test_data6[col_mar]
    test_data6 = test_data6[['Cliente']+entregables]
    
    
    return test_data6

#%
    
