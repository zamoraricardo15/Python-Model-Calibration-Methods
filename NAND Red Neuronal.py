import numpy as np
import pandas as pd
import math

################################# Ejercicio 3 #################################

def sigmoidea(pesos, predictoras):
  x = 0
  for i in range(len(predictoras)):
    x += pesos[i] * predictoras[i]
  
  x -= pesos[len(pesos) - 1]
  return (1 / (1 + math.exp(-(x))))

def NAND(datos, resultado, pesos, ACTIVACION, corte = 0.5):
  calc = []
  for i in range(datos.shape[0]):
    calc.append(ACTIVACION(pesos, datos.iloc[i, :].values))
  
  return(pd.DataFrame({
    'res': calc, 
    'pred': [1 if a >= corte else 0 for a in calc], 
    'real': resultado
  }))

x = pd.DataFrame({'X1': [0, 1, 0, 1], 'X2': [0, 0, 1, 1]})
z = [1, 1, 1, 0]
NAND(x, z, [-0.5, -0.5, -0.5], sigmoidea) # El utlimo valor corresponde al umbral

################################# Ejercicio 4 #################################

def tan_hiperbolica(pesos, predictoras):
  x = 0
  for i in range(len(predictoras)):
    x += pesos[i] * predictoras[i]
  
  x -= pesos[len(pesos) - 1]
  return ((2 / (1 + math.exp(-2 * x))) - 1)

def ECM(pred, real):
  ecm = [(pred[i] - real[i])**2 for i in range(len(pred))]
  return(sum(ecm) / len(pred))

x = pd.DataFrame({'X1': [1, 1, 1, 1], 'X2': [0, 0, 1, 1], 'X3': [0, 1, 0, 1]})
z = [1, 1, 1, 0]

resultados = pd.DataFrame()
for w1 in np.arange(-10, 11)/10:
  for w2 in np.arange(-10, 11)/10:
    for w3 in np.arange(-10, 11)/10:
      for umbral in np.arange(0, 11)/10:
        aux = NAND(x, z, [w1, w2, w3, umbral], tan_hiperbolica, corte = 0)
        e = ECM(aux["pred"].values, aux["real"].values)
        nuevo = pd.DataFrame({"w1": [w1], "w2": [w2], "w3": [w3], "umbral": [umbral], "ECM": [e]})
        resultados = resultados.append(nuevo)
        
resultados.loc[resultados["ECM"] == 0, :]



