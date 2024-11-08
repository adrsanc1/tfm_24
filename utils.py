import pandas as pd
import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE

def load(nombre_archivo):
    # Carga el archivo en un DataFrame de pandas
    df_meta = pd.read_csv(nombre_archivo)
    return df_meta

def balanceo_total(dataframe,feature,numero):
    df0 = dataframe.copy()
    df = pd.concat([df0[feature],df0.iloc[:,5:]],axis=1)

    n_neighbors = df[feature].value_counts().min() -1

    if n_neighbors <= numero:
        adasyn = ADASYN (n_neighbors=n_neighbors, random_state=2)
    else:
        adasyn = ADASYN (random_state=2)
        
    X = df.drop(feature, axis=1)
    y = df[feature]
    
    X_adasyn, y_adasyn = adasyn.fit_resample(X,y)
    # Generar muestras aleatorias de la clase minoritaria para igualar el número de muestras de la clase mayoritaria
    minoritario = pd.concat([pd.Series(y_adasyn, name=feature), X_adasyn],axis=1)

    return minoritario

def completar(dataframe,feature, minoritario, df1):
    df0 = dataframe.copy()

    df_balanced = pd.concat([df0,minoritario.iloc[len(df1):,:]])
    df_balanced.reset_index(drop=True,inplace=True)

    new_features = df1.columns[:5]
    for name in new_features:
        df_balanced.loc[len(df0):,name] = df1[df1[feature]==minoritario.iloc[len(minoritario)-1,0]][name].min()

    return df_balanced


def balanceo_smote(dataframe,feature):
    df0 = dataframe.copy()
    df = pd.concat([df0[feature],df0.iloc[:,5:]],axis=1)

    smote = SMOTE ( random_state=2)
        
    X = df.drop(feature, axis=1)
    y = df[feature]
    
    X_adasyn, y_adasyn = smote.fit_resample(X,y)
    # Generar muestras aleatorias de la clase minoritaria para igualar el número de muestras de la clase mayoritaria
    minoritario = pd.concat([pd.Series(y_adasyn, name=feature), X_adasyn],axis=1)

    return minoritario

def table_pivot(dataframe,list, index, columns):
    return pd.pivot_table(dataframe[list], index=index, columns=columns, aggfunc=len, fill_value=0)


class Homocedasticity:
    def __init__(self, dataframe):
        # Calcula la varianza de cada columna
        self.varianzas = dataframe.var()

    def varianza(self):
        return self.varianzas
        
    def verify(self):
        # Verifica la homocedasticidad comparando las varianzas
        homocedasticidad = np.allclose(self.varianzas, self.varianzas.mean())
        print("Homocedasticidad de las columnas:", homocedasticidad)
        
    def plot(self):
        # Gráfico de caja para visualizar la varianza
        sns.boxplot(data=pd.DataFrame(self.varianzas.values.reshape(1,len(self.varianzas)),columns=self.varianzas.index), orient="h")
        plt.title("Varianza de las columnas")
        plt.xlabel("Valor")
        plt.show()

    def boxplot(self):
        # Gráfico de caja para visualizar la varianza
        sns.boxplot(pd.DataFrame(self.varianzas.values.reshape(1, len(self.varianzas)),
                                 columns=self.varianzas.index).T)

    def dataframe(self):
        return pd.DataFrame(self.varianzas.values.reshape(1, len(self.varianzas)),
                            columns=self.varianzas.index)


def iteracion_variables(dataframe ,type1, type2, features, lote):
    df = dataframe.copy()
    #Varianza por filas
    df1 = df.loc[:,features][df.Desc == type1]
    varianza1 = df1.var(axis=1) 

    df2 = df.loc[:,features][df.Desc == type2]
    varianza2 = df2.var(axis=1)

    #Selecciono filas en función del percentil de todas las filas
    filas1 = [k for k,v in zip(varianza1.index,varianza1) if np.percentile(varianza1, 25)<v<np.percentile(varianza1, 75)]
    filas2 = [k for k,v in zip(varianza2.index,varianza2) if np.percentile(varianza2, 25)<v<np.percentile(varianza2, 75)]

    #Varianza para cada feature
    varianza_1 = df[df.Desc == type1].loc[filas1,features].var(axis=0)
    varianza_2 = df[df.Desc == type2].loc[filas2,features].var(axis=0)

    diferencia = [abs(x-y) for x, y in zip(varianza_1,varianza_2)]
    features = [(key,value) for key, value in zip(varianza_1.index,diferencia)]
    features_sorted = sorted(features, key=lambda x: x[1], reverse=True) # De mayor diferencia a menor
    variables = [x[0] for x in features_sorted][:lote]
    
    return variables, features_sorted

