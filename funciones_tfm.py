#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mannwhitneyu
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


def load_directory(nombre_archivo, directorio):
    directorio_inicial=os.getcwd()
    # Carga el archivo en un DataFrame de pandas
    os.chdir(directorio)
    df_meta = pd.read_csv(nombre_archivo, delimiter='\t', index_col='id')
    os.chdir(directorio_inicial)
    return df_meta


# In[ ]:


def load(nombre_archivo):
    # Carga el archivo en un DataFrame de pandas
    df_meta = pd.read_csv(nombre_archivo)
    return df_meta


# In[ ]:


def save(dataframe, nombre_archivo):
    dataframe.to_csv(nombre_archivo, index=False)


# In[ ]:


def floating(dataframe):
    df=dataframe.copy()
    #Cambio columnas a float
    for col in df.columns:
        df[col]=df[col].astype(float)
    return df


# In[ ]:


def scaler(dataframe):
    df=dataframe.copy()
    # Inicializar el escalador MinMax
    scaler = MinMaxScaler()
    
    # Normalizar los datos del DataFrame
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# In[ ]:


def df_concat(dataframe1,dataframe2):
    df=pd.concat([dataframe1,dataframe2],axis=1)
    return df



# In[ ]:


def u_man(dataframe1,dataframe2):
    # Lista para almacenar los p-valores de la prueba U de Mann-Whitney para cada par de columnas
    p_valores = []
    
    # Iterar sobre las columnas de ambos dataframes y calcular los p-valores para cada par de columnas
    for columna_df1, columna_df2 in zip(dataframe1.T.values, dataframe2.T.values):
        p_valor = mannwhitneyu(columna_df1, columna_df2).pvalue
        p_valores.append(p_valor)
    
    
    diccionario={}
    # Imprimir los p-valores obtenidos para cada par de columnas
    for i, p_valor in zip(dataframe1.columns,p_valores):
        diccionario[i]=p_valor
        print(f"P-valor para la columna {i}: {p_valor}")
    print("P-valor de la prueba U de Mann-Whitney:", p_valor)

    return diccionario


# In[ ]:


def table_pivot(dataframe,list, index, columns):
    return pd.pivot_table(dataframe[list], index=index, columns=columns, aggfunc=len, fill_value=0)


# In[ ]:


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


# In[ ]:


def matrix_correlation(dataframe_cancer, dataframe_sane):
    # Crear una única figura para todos los subgráficos
    #fig, axs = plt.subplots(1, len(df.Fluid.unique()), figsize=(2, 100), sharey=True)
    
    plt.figure(figsize=(2, 30))
    # Filtrar los DataFrames para el flu actual
    df_cancer = dataframe_cancer.copy()
    df_nocancer = dataframe_sane.copy()
    #df_cancer.reset_index(drop=True,inplace=True)
    # Calcular la correlación entre los dos DataFrames
    correlacion = df_cancer.corrwith(df_nocancer)
    
    stat=correlacion.describe()
    
    print(stat)
    #correlacion_filtrada=correlacion[abs(correlacion) < (stat.loc['25%'])]
    
    # Visualizar el heatmap como un subgráfico en la posición actual
    sns.heatmap(correlacion.to_frame(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=False)
    
    plt.show()


# In[ ]:


class Graph_heatmap:
    def __init__(self, dataframe_cancer, dataframe_sane):
        self.df_cancer=dataframe_cancer.copy()
        self.df_nocancer=dataframe_sane.copy()

        # Calcular la correlación entre los dos DataFrames
        correlacion = self.df_cancer.corrwith(self.df_nocancer)
        
        stat=abs(correlacion).describe()
        #Selecciono los que menos correlación presentan entre cáncer y no cáncer
        self.columnas_filtradas=correlacion[abs(correlacion) < (stat.loc['75%'])].index #((stat.loc['25%'])+(stat.loc['min']))/0.1].index
        #columnas_filtradas=correlacion[abs(correlacion) > (stat.loc['75%'])].index
        #df.reset_index(drop=True,inplace=True)
        print(len(self.columnas_filtradas))

    def heatmap(self):
        def plt_heatmap(df,title):
            ##CANCER
            ##
            # Convertir el DataFrame en una matriz NumPy
            matriz_valores = df[self.columnas_filtradas].to_numpy()
            
            # Mostrar la matriz como una imagen
            plt.figure(figsize=(50, 100))
            plt.imshow(matriz_valores, cmap='viridis', aspect='auto')
            
            # Mostrar los nombres de las variables en los ejes
            plt.yticks(ticks=np.arange(0, len(df[self.columnas_filtradas])),
                       labels=df.index)
            plt.xticks(ticks=np.arange(0, len(df[self.columnas_filtradas].columns)),
                       labels=df[self.columnas_filtradas].columns, rotation=90)
            
            # Configurar los ejes
            plt.colorbar(label='Valor de la Variable')
            plt.xlabel('Variable')
            plt.ylabel('Fila')
            plt.title(f'{title}')
            
            # Mostrar la imagen
            plt.show()
        
        plt_heatmap(self.df_cancer, 'Cancer como Imagen')
        plt_heatmap(self.df_nocancer, 'Cancer como Imagen')
        

    def clustermap(self):
        plt.figure(figsize=(50, 100))
        sns.clustermap(self.df_cancer[self.columnas_filtradas])#,row_cluster=False
        plt.title(f'Cancer como una Imagen')
        plt.show()
        
        plt.figure(figsize=(50, 100))
        sns.clustermap(self.df_nocancer[self.columnas_filtradas])
        plt.title(f'No Cancer como  una Imagen')
        plt.show()

    def filter_columns(self):
        return self.columnas_filtradas


# In[ ]:


def cross_validation(dataframe, n_splits=10):
    cv = KFold(n_splits = n_splits, shuffle = True, random_state=0) # 
    
    total_scores = []
    for i in range(2, 35):
       model = DecisionTreeClassifier(criterion='gini', max_depth=i,random_state=0)
       #model =RandomForestClassifier(n_estimators=10)
       fold_accuracy = []
       for train_fold, test_fold in cv.split(dataframe):
          # División train test aleatoria
          f_train = dataframe.iloc[train_fold,5:]
          f_test = dataframe.iloc[test_fold,5:]
          # entrenamiento y ejecución del modelo
          model.fit( X = f_train.drop(['Desc'], axis=1), y = f_train['Desc'])
          y_pred = model.predict(X = f_test.drop(['Desc'], axis = 1))
          # evaluación del modelo
          acc = accuracy_score(f_test['Desc'], y_pred)
          fold_accuracy.append(acc)
       total_scores.append(sum(fold_accuracy)/len(fold_accuracy))
    
    
    max_depth = np.argmax(total_scores) + 2 # +2 porque range(2, 50) y argmax 
    # devuelve el índice del vector cuyo valor es máximo, y ese vector está indexado comenzando en 0
    print ('Max Depth Value ' + str(max(total_scores)) +" (" + str(max_depth) + ")")
    
    plt.plot(range(1,len(total_scores)+1), total_scores, 
             marker='o')
    plt.ylabel('ACC')   
    
    plt.show() 

