#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[11]:


# Carga de datos
df = pd.read_csv('wineData.txt', sep=',')
df.head()

# Nombres de las columnas
nombres_columnas = ['Class','Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# Asignación de nuevos nombres de columnas
df.columns = nombres_columnas

#ver el archivo 
print (df[:20])


# In[46]:


#Exercici 1 y 2 
#Exercici 1 Crea almenys dos models de regressió diferents per intentar predir per intentar predir el millor les classes de l'arxiu adjunt. 
#Exercici 2: Compara els models de classificació utilitzant la precisió (accuracy), una matriu de confusió i d’altres mètriques més avançades.(Precision, Recall, F1 Score, Sensibilidad, Especificidad)
#DESARROLLO:
#Exercici 1: primer modelo: KNN
#Separa los datos en características (atributos) y etiquetas (clase). En este caso, la clase es la columna "Class" y las características son todas las demás columnas. Puedes hacerlo con las siguientes líneas de código:
X = df.drop('Class', axis=1)
y = df['Class']

#Separa los datos en conjuntos de entrenamiento y prueba. En este caso, utilizaremos un 70% de los datos para entrenamiento y el 30% para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Normaliza los datos para que todas las características tengan la misma escala
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Entrena el modelo KNN utilizando el conjunto de entrenamiento. En este caso, utilizaremos un valor de k=3. 
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#Haz predicciones sobre el conjunto de prueba utilizando el modelo entrenado. 
y_pred = knn.predict(X_test)

#EXERCICI 2 Primer modelo KNN:
#Evalúa el rendimiento del modelo calculando la precisión (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo KNN: {accuracy:.2f}')

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:')
print(confusion)

from sklearn.metrics import precision_score, recall_score, f1_score

# Calcular precision, recall y F1 Score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Calcular sensibilidad y especificidad a partir de la matriz de confusión
sensitivity = confusion[0,0]/(confusion[0,0]+confusion[0,1])
specificity = confusion[1,1]/(confusion[1,0]+confusion[1,1])

print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
print("Sensibilidad: {:.2f}".format(sensitivity))
print("Especificidad: {:.2f}".format(specificity))


# In[44]:


#EXERCICI 3: Entrena’ls usant diferents paràmetres que admeten per tal de millorar-ne la predicció.
# Separa los datos en características (atributos) y etiquetas (clase)
X = df.drop('Class', axis=1)
y = df['Class']

# Separa los datos en conjuntos de entrenamiento y prueba (70% para entrenamiento y 30% para prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normaliza los datos para que todas las características tengan la misma escala
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrena el modelo KNN utilizando el conjunto de entrenamiento. En este caso, utilizaremos un valor de k=5. 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Haz predicciones sobre el conjunto de prueba utilizando el modelo entrenado. 
y_pred = knn.predict(X_test)

# Evalúa el rendimiento del modelo calculando la precisión (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo KNN: {accuracy:.2f}')

# Calcula la matriz de confusión y muestra los resultados
confusion = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:')
print(confusion)

# Calcula precision, recall y F1 Score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Calcula sensibilidad y especificidad a partir de la matriz de confusión
sensitivity = confusion[0,0]/(confusion[0,0]+confusion[0,1])
specificity = confusion[1,1]/(confusion[1,0]+confusion[1,1])

# Muestra los resultados de las métricas
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
print("Sensibilidad: {:.2f}".format(sensitivity))
print("Especificidad: {:.2f}".format(specificity))


# In[64]:


#Exercici 5
#Aplica algun procés d'enginyeria per millorar els resultats (normalització, estandardització, mostreig...)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Separar características y etiquetas
X = df.drop('Class', axis=1)
y = df['Class']

# Aplicar la selección de características
best_features = SelectKBest(score_func=chi2, k=2)
X_new = best_features.fit_transform(X, y)

# Imprimir las características seleccionadas
print(X.columns[best_features.get_support()])

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Separar los datos en conjuntos de entrenamiento y prueba utilizando las características seleccionadas
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)

# Entrenar el modelo KNN utilizando las dos características seleccionadas
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba utilizando el modelo entrenado
y_pred = knn.predict(X_test)

# Evaluar el rendimiento del modelo utilizando la precisión (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo KNN utilizando las dos características seleccionadas: {accuracy:.2f}')
print("El proceso utilizado, no mejoró la precisión del modelo")


# In[39]:


#DESARROLLO:
#Exercici 1: Segundo modelo: SVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Separar los datos en características (X) y etiquetas (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# crear un clasificador SVM y ajustarlo a los datos de entrenamiento
svm_clf = SVC(kernel='linear', C=1, random_state=42)
svm_clf.fit(X_train, y_train)

# Hacer predicciones sobre los datos de prueba
y_pred_svm = svm_clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred_svm)
print('Precisión del modelo: {:.2f}%'.format(accuracy*100))

# calcular la matriz de confusión
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("Matriz de Confusión:\n", cm_svm)

# calcular la sensibilidad y especificidad
if cm_svm.shape[0] == 2:
    tn, fp, fn, tp = cm_svm.ravel()
    sens = tp / (tp + fn)
    espec = tn / (tn + fp)
    print("Sensibilidad:", sens)
    print("Especificidad:", espec)
elif cm_svm.shape[0] > 2:
    sens = []
    espec = []
    for i in range(cm_svm.shape[0]):
        tn = np.sum(np.delete(np.delete(cm_svm, i, 0), i, 1))
        fp = np.sum(np.delete(cm_svm, i, 0)[:, i])
        fn = np.sum(np.delete(cm_svm, i, 1)[i, :])
        tp = cm_svm[i, i]
        sens.append(tp / (tp + fn))
        espec.append(tn / (tn + fp))
    print("Sensibilidad por clase:", sens)
    print("Especificidad por clase:", espec)
else:
    print("La matriz de confusión no es válida.")
    
# calcular el F1 score
f1_svm = classification_report(y_test, y_pred_svm)
print("F1 score:\n", f1_svm)


# In[ ]:


#RESPUESTA ejercicio 2: Ambos modelos tienen una precisión similar. Sin embargo, el modelo SVC proporciona información adicional como la sensibilidad y la especificidad por clase, así como la precisión, el recall y el F1-score para cada clase individual.


# In[61]:


#EXERCICI 3 DE MODELO SVC
#Entrena’ls usant els diferents paràmetres que admeten per tal de millorar-ne la predicció.

from sklearn.model_selection import GridSearchCV

# Parámetros a evaluar
param_grid = {'C': [0.1, 1, 10, 100]}

# Clasificador SVM
svm_clf = SVC(kernel='linear', random_state=42)

# Realizar la búsqueda de cuadrícula
grid_search = GridSearchCV(svm_clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo y la precisión correspondiente
best_svm_clf = grid_search.best_estimator_
best_accuracy = grid_search.best_score_

from sklearn.metrics import accuracy_score

y_pred = best_svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Precisión del modelo SVM: {:.2f}%".format(accuracy * 100))
print("El cambio de parámetros, no mejoró la precisión del modelo")


# In[60]:


#Exercici 4: Compara el seu rendiment fent servir l’aproximació traint/test o cross-validation.
    
from sklearn.model_selection import cross_val_score

# Modelo 1 - KNN
knn = KNeighborsClassifier(n_neighbors=3)
scores_knn = cross_val_score(knn, X, y, cv=5)
print(f"Precision media del modelo KNN: {scores_knn.mean():.2f}")

# Modelo 2 - SVM
svm_clf = SVC(kernel='linear', C=1, random_state=42)
scores_svm = cross_val_score(svm_clf, X, y, cv=5)
print(f"Precision media del modelo SVM: {scores_svm.mean():.2f}")
print("En este caso, la precisión media del modelo KNN es 0.69, lo que significa que, en promedio, el modelo KNN acertó en un 69% de las predicciones realizadas. Por otro lado, la precisión media del modelo SVM es 0.96, lo que indica que, en promedio, el modelo SVM tuvo una tasa de acierto del 96% en sus predicciones.")


# In[63]:


#Exercici 5-MODELO 2
#Aplica algun procés d'enginyeria per millorar els resultats (normalització, estandardització, mostreig...)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Separar los datos en características (X) y etiquetas (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar los datos con MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# crear un clasificador SVM y ajustarlo a los datos de entrenamiento normalizados
svm_clf = SVC(kernel='linear', C=1, random_state=42)
svm_clf.fit(X_train, y_train)

# Hacer predicciones sobre los datos de prueba normalizados
y_pred_svm = svm_clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred_svm)
print('Precisión del modelo: {:.2f}%'.format(accuracy*100))
print("El método de normalización con MinMaxScaler, no mejoró la precisión del modelo")


# In[ ]:




