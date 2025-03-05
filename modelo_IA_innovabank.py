# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:58:08 2025

@author: monts
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#datos inventados
np.random.seed(42)
ingresos = np.random.randint(2000,100000, 1000)
gastos = np.random.randint(1000, 9000, 1000)
historial_crediticio = np.random.randint(300, 850, 1000) #esta es la score de credito
solvente = np.where((ingresos - gastos > 2000) & (historial_crediticio > 600), 1 , 0)

#este es el dataframe
df =pd.DataFrame({
    "ingresos": ingresos,
    "gastos": gastos,
    "historial_crediticio": historial_crediticio,
    "solvente": solvente
    })

#aca preparamos los datos 

x = df [["ingresos","gastos","historial_crediticio",]]
y = df["solvente"]

#dividimos el entrenamiento 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#creamos el modelo 
modelo = LogisticRegression()

#entrenamos el modelo
modelo.fit(x_train, y_train)

#hacemos predicciones con los datos de prueba 
y_pred = modelo.predict(x_test)

#evaluamos la precision 
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')

#visualizamos los resultados
# Crear la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Graficarla
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()
