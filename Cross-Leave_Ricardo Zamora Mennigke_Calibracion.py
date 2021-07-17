# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:07:48 2020

@author: rzamoram
"""

##Tarea 3

#Pregunta 1
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas import DataFrame
from matplotlib import colors as mcolors
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier







pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/OneDrive - Intel Corporation/Documents/Machine Learning/Métodos Supervisados con Python/Clase 01")
print(os.getcwd())
datos = pd.read_csv('tumores.csv',delimiter=',',decimal=".")
datos['imagen'] = datos['imagen'].astype('category')
print(datos.shape)
print(datos.head())
print(datos.info())



def distribucion_variable_predecir(data:DataFrame,variable_predict:str):
    colors = list(dict(**mcolors.CSS4_COLORS))
    df = pd.crosstab(index=data[variable_predict],columns="valor") / data[variable_predict].count()
    fig = plt.figure(figsize=(10,9))
    g = fig.add_subplot(111)
    countv = 0
    titulo = "Distribución de la variable %s" % variable_predict
    for i in range(df.shape[0]):
        g.barh(1,df.iloc[i],left = countv, align='center',color=colors[11+i],label= df.iloc[i].name)
        countv = countv + df.iloc[i]
    vals = g.get_xticks()
    g.set_xlim(0,1)
    g.set_yticklabels("")
    g.set_title(titulo)
    g.set_ylabel(variable_predict)
    g.set_xticklabels(['{:.0%}'.format(x) for x in vals])
    countv = 0 
    for v in df.iloc[:,0]:
        g.text(np.mean([countv,countv+v]) - 0.03, 1 , '{:.1%}'.format(v), color='black', fontweight='bold')
        countv = countv + v
    g.legend(loc='upper center', bbox_to_anchor=(1.08, 1), shadow=True, ncol=1)

distribucion_variable_predecir(datos,"tipo")

def indices_general(MC, nombres = None):
    precision_global = np.sum(MC.diagonal()) / np.sum(MC)
    error_global = 1 - precision_global
    precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
    precision_positiva = MC[1][1]/(MC[1][1] + MC[1][0])
    precision_negativa = MC[0][0]/(MC[0][0] + MC[0][1])
    falsos_positivos = 1 - precision_negativa
    falsos_negativos = 1 - precision_positiva
    asertividad_positiva = MC[1][1]/(MC[0][1] + MC[1][1])
    asertividad_negativa = MC[0][0]/(MC[0][0] + MC[1][0])
    if nombres!=None:
        precision_categoria.columns = nombres
    return {"Matriz de Confusión":MC, 
            "Precisión Global":precision_global, 
            "Error Global":error_global, 
            "Precisión por categoría":precision_categoria,
            "Precision Positiva (PP)": precision_positiva, 
            "Precision Negativa (PN)":precision_negativa, 
            "Falsos Positivos(FP)": falsos_positivos,
            "Falsos Negativos (FN)": falsos_negativos,
            "Asertividad Positiva (AP)": asertividad_positiva,
            "Asertividad Negativa (NP)": asertividad_negativa}
    
def poder_predictivo_categorica(data:DataFrame, var:str, variable_predict:str):
    df = pd.crosstab(index= data[var],columns=data[variable_predict])
    df = df.div(df.sum(axis=1),axis=0)
    titulo = "Distribución de la variable %s según la variable %s" % (var,variable_predict)
    g = df.plot(kind='barh',stacked=True,legend = True, figsize = (10,9), \
                xlim = (0,1),title = titulo, width = 0.8)
    vals = g.get_xticks()
    g.set_xticklabels(['{:.0%}'.format(x) for x in vals])
    g.legend(loc='upper center', bbox_to_anchor=(1.08, 1), shadow=True, ncol=1)
    for bars in g.containers:
        plt.setp(bars, width=.9)
    for i in range(df.shape[0]):
        countv = 0 
        for v in df.iloc[i]:
            g.text(np.mean([countv,countv+v]) - 0.03, i , '{:.1%}'.format(v), color='black', fontweight='bold')
            countv = countv + v
            
def poder_predictivo_numerica(data:DataFrame, var:str, variable_predict:str):
    sns.FacetGrid(data, hue=variable_predict, height=6).map(sns.kdeplot, var, shade=True).add_legend()




X = datos.iloc[:,1:17] 
print(X.head())
y = datos.iloc[:,17:18] 
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

###2
error_tt = []
for i in range(0, 6):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)
    knn = KNeighborsClassifier(n_neighbors = 50)
    knn.fit(X_train, y_train.values.ravel())
    error_tt.append(1 - knn.score(X_test, y_test))
  
plt.figure(figsize=(15,10))
plt.plot(error_tt, 'o-', lw = 2)
plt.xlabel("Número de Iteración", fontsize = 15)
plt.ylabel("Error Cometido %", fontsize = 15)
plt.title("Variación del Error", fontsize = 20)
plt.grid(True)
plt.legend(['Training Testing'], loc = 'upper right', fontsize = 15)

################
error_tc = []

for i in range(0, 6):
    knn = KNeighborsClassifier(n_neighbors = 50)
    knn.fit(X, y.values.ravel())
    
    error_tc.append(1 - knn.score(X, y))

plt.figure(figsize=(15,10))
plt.plot(error_tt, 'o-', lw = 2)
plt.plot(error_tc, 'o-', lw = 2)
plt.xlabel("Número de Iteración", fontsize = 15)
plt.ylabel("Error Cometido %", fontsize = 15)
plt.title("Variación del Error", fontsize = 20)
plt.grid(True)
plt.legend(['Training Testing', 'Tabla Completa'], loc = 'upper right', fontsize = 15)


###3
error_cv = []

for i in range(0, 6):
    kfold = KFold(n_splits = 10, shuffle = True)
    error_folds = []

    for train, test in kfold.split(X, y):
        knn = KNeighborsClassifier(n_neighbors = 50)
        knn.fit(X.iloc[train], y.iloc[train].values.ravel())
        error_folds.append((1 - knn.score(X.iloc[test], y.iloc[test])))
        
    error_cv.append(np.mean(error_folds))

plt.figure(figsize=(15,10))
plt.plot(error_tt, 'o-', lw = 2)
plt.plot(error_tc, 'o-', lw = 2)
plt.plot(error_cv, 'o-', lw = 2)
plt.xlabel("Número de Iteración", fontsize = 15)
plt.ylabel("Error Cometido", fontsize = 15)
plt.title("Variación del Error", fontsize = 20)
plt.grid(True)
plt.legend(['K-Fold CV', 'Training Testing', 'Tabla Completa'], loc = 'upper right', fontsize = 15)

error_tc = []

for i in range(0, 6):
    knn = KNeighborsClassifier(n_neighbors = 50)
    knn.fit(X, y.values.ravel())
    
    error_tc.append(1 - knn.score(X, y))

plt.figure(figsize=(15,10))
plt.plot(error_tt, 'o-', lw = 2)
plt.plot(error_tc, 'o-', lw = 2)
plt.xlabel("Número de Iteración", fontsize = 15)
plt.ylabel("Error Cometido %", fontsize = 15)
plt.title("Variación del Error", fontsize = 20)
plt.grid(True)
plt.legend(['Training Testing', 'Tabla Completa'], loc = 'upper right', fontsize = 15)



#Ejercicio 2
error_cv = []

for i in range(0, 9):
    kfold = KFold(n_splits = 10, shuffle = True)
    error_folds = []

    for train, test in kfold.split(X, y):
        knn = RandomForestClassifier(n_estimators=10)
        knn.fit(X.iloc[train], y.iloc[train].values.ravel())
        error_folds.append((1 - knn.score(X.iloc[test], y.iloc[test])))
        
    error_cv.append(np.mean(error_folds))

plt.figure(figsize=(15,10))
#plt.plot(error_tt, 'o-', lw = 2)
#plt.plot(error_tc, 'o-', lw = 2)
#plt.plot(error_loo, 'o-', lw = 2)
plt.plot(error_cv, 'o-', lw = 2)
plt.xlabel("Número de Iteración", fontsize = 15)
plt.ylabel("Error Cometido", fontsize = 15)
plt.title("Variación del Error", fontsize = 20)
plt.grid(True)
plt.legend(['Training Testing', 'Tabla Completa', 'Dejando Uno Fuera', 'K-Fold CV'], loc = 'upper right', fontsize = 15)



n = datos.shape[0]
error_loo = []

for i in range(0, 5):
    error_i = 0
    
    for j in range(0, n): 
        X_train = X.drop(j, axis = 0)
        X_test = X.iloc[j, :]
        y_train = y.drop(j, axis = 0)
        y_test = y.iloc[j]
        
        knn = KNeighborsClassifier(n_neighbors = 50)
        knn.fit(X_train, y_train.values.ravel())
        prediccion = knn.predict(pd.DataFrame(X_test).T)
               
        if(prediccion != y_test[0]):
            error_i = error_i + 1
        
    error_loo.append(error_i / n)

plt.figure(figsize=(15,10))
plt.plot(error_tt, 'o-', lw = 2)
plt.plot(error_tc, 'o-', lw = 2)
plt.plot(error_loo, 'o-', lw = 2)
plt.xlabel("Número de Iteración", fontsize = 15)
plt.ylabel("Error Cometido %", fontsize = 15)
plt.title("Variación del Error", fontsize = 20)
plt.grid(True)
plt.legend(['Training Testing', 'Tabla Completa', 'Dejando Uno Fuera'], loc = 'upper right', fontsize = 15)

#Ejercicio 3

error_cv = []

for i in range(0, 6):
    kfold = KFold(n_splits = 10, shuffle = True)
    error_folds = []

    for train, test in kfold.split(X, y):
        knn = KNeighborsClassifier(n_neighbors = 50)
        knn.fit(X.iloc[train], y.iloc[train].values.ravel())
        error_folds.append((1 - knn.score(X.iloc[test], y.iloc[test])))
        
    error_cv.append(np.mean(error_folds))

plt.figure(figsize=(15,10))
plt.plot(error_tt, 'o-', lw = 2)
plt.plot(error_tc, 'o-', lw = 2)
#plt.plot(error_loo, 'o-', lw = 2)
plt.plot(error_cv, 'o-', lw = 2)
plt.xlabel("Número de Iteración", fontsize = 15)
plt.ylabel("Error Cometido", fontsize = 15)
plt.title("Variación del Error", fontsize = 20)
plt.grid(True)
plt.legend(['Training Testing', 'Tabla Completa', 'Dejando Uno Fuera', 'K-Fold CV'], loc = 'upper right', fontsize = 15)

##Ejercicio 2
instancia_kfold = KFold(n_splits=10)
porcentajes = cross_val_score(RandomForestClassifier(n_estimators=10, random_state=0, criterion='gini'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_gini = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))

instancia_kfold = KFold(n_splits=10)
porcentajes = cross_val_score(RandomForestClassifier(n_estimators=10, random_state=0, criterion='entropy'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_entropy = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,5))
alto = [res_gini, res_entropy]
barras = ('gini','entropy')
y_pos = np.arange(len(barras))
plt.bar(y_pos, alto, color=['red', 'green'])
plt.xticks(y_pos, barras)
plt.show()

##Ejercicio 3
instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(KNeighborsClassifier(n_neighbors=5, algorithm='auto'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_auto = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_ball = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_kd = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(KNeighborsClassifier(n_neighbors=5, algorithm='brute'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_brute = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))

plt.figure(figsize=(8,5))
alto = [res_auto,res_ball,res_kd,res_brute]
barras = ('sigmoid','rbf','poly','linear')
y_pos = np.arange(len(barras))
plt.bar(y_pos, alto, color=['red', 'green', 'blue', 'cyan'])
plt.xticks(y_pos, barras)
plt.show()

##Ejercicio 4
from sklearn.neighbors import KNeighborsClassifier
instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(KNeighborsClassifier(n_neighbors=5), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_knn = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


from sklearn.tree import DecisionTreeClassifier
instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(DecisionTreeClassifier(), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_arbol = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


from sklearn.ensemble import RandomForestClassifier
instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(RandomForestClassifier(n_estimators=5, random_state=0, criterion='gini'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_bosques = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))



from sklearn.ensemble import AdaBoostClassifier
instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(AdaBoostClassifier(n_estimators=5), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_potenciacion = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


from sklearn.ensemble import GradientBoostingClassifier
instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(GradientBoostingClassifier(n_estimators=5), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_xg_potenciacion = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))



instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(SVC(kernel='rbf', gamma = 'scale'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_svm = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


from sklearn.neural_network import MLPClassifier
instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(MLPClassifier(solver='lbfgs'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_redes_MLP = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


from sklearn.naive_bayes import GaussianNB
instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(GaussianNB(), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_bayes = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 'auto'), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_dis_lineal = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(QuadraticDiscriminantAnalysis(), X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_dis_cuadratico = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))


import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(15,5))
alto = [res_knn,res_arbol,res_bosques,res_potenciacion,res_xg_potenciacion,res_svm,res_redes_MLP,res_bayes,res_dis_lineal,res_dis_cuadratico]
barras = ('KNN','Árbol','Bosques','Potenciación','XGPotenciación','SVM','Redes','Bayes','Dis-Lineal','Dis-Cuadrático')
y_pos = np.arange(len(barras))
plt.bar(y_pos, alto,color = ["#67E568","#257F27","#08420D","#FFF000","#FFB62B","#E56124","#E53E30","#7F2353","#F911FF","#9F8CA6", "#2ecc71"])
plt.xticks(y_pos, barras)
plt.show()



from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim = 16, activation = 'relu')) 
model.add(Dense(10, activation = 'relu'))  
model.add(Dense(10, activation = 'relu'))  
model.add(Dense(5, activation = 'sigmoid')) 
model.add(Dense(1, activation = 'sigmoid')) 
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
scores = model.evaluate(X, y, verbose=0)

instancia_kfold = KFold(n_splits=5)
porcentajes = cross_val_score(scores, X, y.iloc[:,0].values, cv=instancia_kfold)
print("Porcentaje de detección por grupo:\n{}".format(porcentajes))
res_keras = porcentajes.mean()
print("Promedio de detección: {:.2f}".format(porcentajes.mean()))















