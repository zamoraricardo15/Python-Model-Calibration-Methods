# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 23:20:41 2020

@author: rzamoram
"""

##Pregunta 1

import numpy as np
import pandas as pd
import random as rd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from matplotlib import colors as mcolors
import seaborn as sns


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
    









from sklearn.ensemble import RandomForestClassifier


def plotROC(real, prediccion, color = "red", label = None):
    fp_r, tp_r, umbral = roc_curve(real, prediccion)
    plt.plot(fp_r, tp_r, lw = 1, color = color, label = label)
    plt.plot([0, 1], [0, 1], lw = 1, color = "black")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curva ROC")
    


X = datos.iloc[:,1:17] 
print(X.head())
y = datos.iloc[:,17:18] 
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)


instancia_svm = SVC(kernel = "rbf",gamma='scale',probability=True)
instancia_svm.fit(X_train, y_train.iloc[:,0].values)
#print("Probabilidad del No y del Si:\n",instancia_svm.predict_proba(X_test))
probabilidad = instancia_svm.predict_proba(X_test)[:, 1]
#print("Probabilidad de Si (o sea del 1):\n",probabilidad)
# Gráfico de la Curva ROC
plt.figure(figsize=(10,10))
plotROC(y_test, probabilidad)

instancia_bosques = RandomForestClassifier(n_estimators = 300, max_features = 3)
instancia_bosques.fit(X_train, y_train.iloc[:,0].values)
# Genera la Curva ROC para Bosques
plt.figure(figsize=(10,10))
plotROC(y_test, instancia_bosques.predict_proba(X_test)[:, 1], color = "blue")


instancia_bosques = RandomForestClassifier(n_estimators = 300, max_features = 3)
instancia_bosques.fit(X_train, y_train.iloc[:,0].values)


from sklearn.neighbors import KNeighborsClassifier
instancia_knn = KNeighborsClassifier(n_neighbors=5)
instancia_knn.fit(X_train, y_train.iloc[:,0].values)
#plt.figure(figsize=(10,10))
#plotROC(y_test, instancia_knn.predict_proba(X_test)[:, 1], color = "blue")


instancia_tree = DecisionTreeClassifier()
instancia_tree.fit(X_train, y_train.iloc[:,0].values)





instancia_ADA = AdaBoostClassifier(n_estimators=5)
instancia_ADA.fit(X_train, y_train.iloc[:,0].values)


from sklearn.ensemble import GradientBoostingClassifier
instancia_XGB = GradientBoostingClassifier(n_estimators=5)
instancia_XGB.fit(X_train, y_train.iloc[:,0].values)






from sklearn.neural_network import MLPClassifier
instancia_classifier = MLPClassifier(solver='lbfgs')
instancia_classifier.fit(X_train, y_train.iloc[:,0].values)


from sklearn.naive_bayes import GaussianNB
instancia_bayes = GaussianNB()
instancia_bayes.fit(X_train, y_train.iloc[:,0].values)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
instancia_lda = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 'auto')
instancia_lda.fit(X_train, y_train.iloc[:,0].values)



from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
instancia_qda = QuadraticDiscriminantAnalysis()
instancia_qda.fit(X_train, y_train.iloc[:,0].values)




plt.figure(figsize=(10,10))
plotROC(y_test, instancia_svm.predict_proba(X_test)[:, 1], label = "SVM")
plotROC(y_test, instancia_bosques.predict_proba(X_test)[:, 1], color = "blue", label= "Bosques Aleatorios")
plotROC(y_test, instancia_knn.predict_proba(X_test)[:, 1], color = "red", label= "KNN")
plotROC(y_test, instancia_tree.predict_proba(X_test)[:, 1], color = "#67E568", label= "Arboles Decision")
#plotROC(y_test, instancia_knn.predict_proba(X_test)[:, 1], color = "#257F27", label= "KNN")
plotROC(y_test, instancia_ADA.predict_proba(X_test)[:, 1], color = "#08420D", label= "ADA")
plotROC(y_test, instancia_XGB.predict_proba(X_test)[:, 1], color = "#FFF000", label= "XGBoosting")
plotROC(y_test, instancia_classifier.predict_proba(X_test)[:, 1], color = "#FFB62B", label= "Redes neuronales")
plotROC(y_test, instancia_bayes.predict_proba(X_test)[:, 1], color = "#E56124", label= "Bayes")
plotROC(y_test, instancia_lda.predict_proba(X_test)[:, 1], color = "#E53E30", label= "LDA")
#plotROC(y_test, instancia_qda.predict_proba(X_test)[:, 1], color = "#7F2353", label= "QDA")
plt.legend(loc = "lower right")



bosques_area = roc_auc_score(y_test, instancia_bosques.predict_proba(X_test)[:, 1])
svm_area = roc_auc_score(y_test, instancia_svm.predict_proba(X_test)[:, 1])
knn_area = roc_auc_score(y_test, instancia_knn.predict_proba(X_test)[:, 1])
tree_area = roc_auc_score(y_test, instancia_tree.predict_proba(X_test)[:, 1])
knn_area = roc_auc_score(y_test, instancia_knn.predict_proba(X_test)[:, 1])
ADA_area = roc_auc_score(y_test, instancia_ADA.predict_proba(X_test)[:, 1])
XGB_area = roc_auc_score(y_test, instancia_XGB.predict_proba(X_test)[:, 1])
red_area = roc_auc_score(y_test, instancia_classifier.predict_proba(X_test)[:, 1])
bayes_area = roc_auc_score(y_test, instancia_bayes.predict_proba(X_test)[:, 1])
lda_area = roc_auc_score(y_test, instancia_lda.predict_proba(X_test)[:, 1])
#qda_area = roc_auc_score(y_test, instancia_qda.predict_proba(X_test)[:, 1])
print("Área bajo la curva ROC en Bosques Aleatorios: {:.3f}".format(bosques_area))
print("Área bajo la curva ROC en KNN: {:.3f}".format(knn_area))
print("Área bajo la curva ROC en Arboles Decision: {:.3f}".format(tree_area))
print("Área bajo la curva ROC en KNN: {:.3f}".format(knn_area))
print("Área bajo la curva ROC en ADA: {:.3f}".format(ADA_area))
print("Área bajo la curva ROC en XGB: {:.3f}".format(XGB_area))
print("Área bajo la curva ROC en red: {:.3f}".format(red_area))
print("Área bajo la curva ROC en bayes: {:.3f}".format(bayes_area))
print("Área bajo la curva ROC en LDA: {:.3f}".format(lda_area))
#print("Área bajo la curva ROC en QDA: {:.3f}".format(qda_area))





from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


##Pregunta 2
import os
import pandas as pd
pasada = os.getcwd()
os.chdir("C:/Users/rzamoram/OneDrive - Intel Corporation/Documents/Machine Learning/Métodos NO Supervisados con Python/Clase 1")
os.getcwd()
ejemplo10 = pd.read_csv("SAheart.csv", delimiter = ';', decimal = ".", header = 0, index_col = 0)
print(ejemplo10.head())
datos = pd.DataFrame(ejemplo10)


def recodificar(col, nuevo_codigo):
  col_cod = pd.Series(col, copy=True)
  for llave, valor in nuevo_codigo.items():
    col_cod.replace(llave, valor, inplace=True)
  return col_cod

datos["famhist"] = recodificar(datos["famhist"], {'Present':1,'Absent':2})
datos["chd"] = recodificar(datos["chd"], {'No':0,'Si':1})
print(datos.head())
print(datos.dtypes)
# Conviertiendo la variables en Dummy
datos_dummy = pd.get_dummies(datos)
print(datos_dummy.head())
print(datos_dummy.dtypes)

X = datos.iloc[:,:8] 
print(X.head())
y = datos.iloc[:,8:9] 
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, random_state = 0)



# bosques aleatorios modelo
instancia_bosques = RandomForestClassifier(n_estimators = 300, max_features = 3)
instancia_bosques.fit(X_train, y_train.iloc[:,0].values)
prediccion = instancia_bosques.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))

print("Probabilidad del No y del Si:\n",instancia_bosques.predict_proba(X_test))
probabilidad = instancia_bosques.predict_proba(X_test)[:, 1]
print("Probabilidad de Si (o sea del 1):\n",probabilidad)




# regla de decision
corte = [0.501, 0.502, 0.503, 0.504, 0.505, 0.506, 0.507, 0.508, 0.509]
for c in corte:
    print("===========================")
    print("Probabilidad de Corte: ",c)
    prediccion = np.where(probabilidad > c, 1, 0)
    # Calidad de la predicción 
    MC = confusion_matrix(y_test, prediccion)
    indices = indices_general(MC,list(np.unique(y)))
    for k in indices:
        print("\n%s:\n%s"%(k,str(indices[k])))



# XGB modelo
instancia_XGB = GradientBoostingClassifier(n_estimators=5)
instancia_XGB.fit(X_train, y_train.iloc[:,0].values)
prediccion = instancia_XGB.predict(X_test)
MC = confusion_matrix(y_test, prediccion)
indices = indices_general(MC,list(np.unique(y)))
for k in indices:
    print("\n%s:\n%s"%(k,str(indices[k])))

#print("Probabilidad del No y del Si:\n",instancia_XGB.predict_proba(X_test))
probabilidad = instancia_XGB.predict_proba(X_test)[:, 1]
#print("Probabilidad de Si (o sea del 1):\n",probabilidad)




# regla de decision
corte = [0.501, 0.502, 0.503, 0.504, 0.505, 0.506, 0.507, 0.508, 0.509, 0.6, 0.601, 0.602, 0.603, 0.604, 0.605, 0.606, 0.607, 0.608, 0.609]
for c in corte:
    print("===========================")
    print("Probabilidad de Corte: ",c)
    prediccion = np.where(probabilidad > c, 1, 0)
    # Calidad de la predicción 
    MC = confusion_matrix(y_test, prediccion)
    indices = indices_general(MC,list(np.unique(y)))
    for k in indices:
        print("\n%s:\n%s"%(k,str(indices[k])))


##Pregunta 3
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


Clase = np.array([1,0,1,0,0,1,1,0,1,1])
Score = np.array([0.8,0.7,0.65,0.6,0.5,0.35,0.3,0.25,0.2,0.1])

# Graficamos ROC con usando roc_curve de sklearn
fp_r, tp_r, umbral = roc_curve(Clase, Score)
plt.figure(figsize=(10,10))
plt.plot(fp_r, tp_r, lw = 1, color = "red")
plt.plot([0, 1], [0, 1], lw = 1, color = "black")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")


# Graficamos puntos con el siguiente algoritmo
i = 1  # Contador
FP_r = -1  # Para que entre al condicional en la primera iteración
TP_r = -1  # Para que entre al condicional en la primera iteración

# linspace genera una sucesión de 201 números del 1 al 0 que equivale a una sucesión del 1 al 0 con paso de 0.005
for Umbral in np.linspace(1, 0, 201):   
    Prediccion = np.where(Score >= Umbral, 1, 0)
    MC = confusion_matrix(Clase, Prediccion)   
    if (FP_r != MC[0, 1] / sum(MC[0, ])) | (TP_r != MC[1, 1] / sum(MC[1, ])):     
            FP_r = MC[0, 1] / sum(MC[0, ])  # Tasa de Falsos Positivos
            TP_r = MC[1, 1] / sum(MC[1, ])  # Tasa de Verdaderos Positivos           
            # Graficamos punto
            plt.plot(FP_r, TP_r, "o", mfc = "none", color = "blue")
            plt.annotate(round(Umbral, 3), (FP_r + 0.01, TP_r - 0.02))     
            # Imprimimos resultado
            print("=====================")
            print("Punto i = ", i, "\n")  
            print("Umbral = T = ", round(Umbral, 3), "\n")
            print("MC =")
            print(MC, "\n")
            print("Tasa FP = ", round(FP_r, 2), "\n")
            print("Tasa TP = ", round(TP_r, 2))     
            i = i + 1

#######
Clase = np.array([1,0,1,0,0,1,1,0,1,1])
Score = np.array([0.8,0.7,0.65,0.6,0.5,0.35,0.3,0.25,0.2,0.1])

fp_r, tp_r, umbral = roc_curve(Clase, Score)
plt.figure(figsize=(10,10))
plt.plot(fp_r, tp_r, lw = 1, color = "red")
plt.plot([0, 1], [0, 1], lw = 1, color = "black")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")

# Aquí se inicializan para que de igual a la corrida a pie
Umbral = min(Score)
Paso = (max(Score) - min(Score)) / 10
N = 10 # ceros
P = 10 # unos
TP = 0 
FP = 0
for i in range(0, 10):    
    if Score[i] > Umbral:
        if Clase[i] == 1:
            TP = TP + 1
        else:
            FP = FP + 1
    else:
        if Clase[i] == 0:
            FP = FP + 1
        else:
            TP = TP + 1
    
    # Graficamos punto
    plt.plot(FP / N, TP / P, "o", mfc = "none", color = "blue")
    plt.annotate(i + 1, (FP / N + 0.01, TP / P - 0.02))    
    Umbral = Umbral + Paso




















