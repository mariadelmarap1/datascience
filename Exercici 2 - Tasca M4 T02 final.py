#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# In[6]:


# llegir el fitxer i assignar-lo a un DataFrame
datosp = pd.read_csv('movies.dat', sep='::', header=None, names=['id', 'title', 'raw_genres'],encoding='latin-1')
datosp[:20]


# In[7]:


# Extreure l'any de llançament i assignar-lo a una nova columna
datosp["year"] = datosp["title"].str.extract(r"\((\d{4})\)")

print(datosp)


# In[8]:


# Eliminar anys de la columna "title"
datosp["title"] = datosp["title"].str.replace(r"\(\d{4}\)","")
print(datosp)


# In[9]:


# Crea una nova columna per a cada gènere
datosp = datosp.join(datosp.pop("raw_genres").str.split("|", expand=True))
print(datosp)


# In[10]:


# Crea una funció per seleccionar el primer gènere
def get_first_genre(row):
    return row.iloc[2]
# Aplica la funció a cada fila i crea una nova columna per al gènere seleccionat
datosp["genre"] = datosp.apply(get_first_genre, axis=1)
print(datosp)


# In[11]:


# Elimina les columnes innecessàries
datosp = datosp.drop(columns=[1, 2,3,4,5])
print(datosp)


# In[12]:


# Eliminar la columna ID
datosp = datosp.drop(columns=["id"])
print(datosp)


# In[13]:


# Eliminar la columna Genre que es año
datosp = datosp.drop(columns=["genre"])
print(datosp)


# In[14]:


# Canviar els noms de les columnes
datosp = datosp.rename(columns={"title": "Nom", "year": "Any", 0: "Genere"})
print(datosp)


# In[15]:


sns.countplot(x=datosp['Genere'])
plt.xlabel("Género de Película") # Nombre del eje x
plt.ylabel("Cantidad de Películas") # Nombre del eje y
plt.title("Proporción de Películas por Género en un Año Específico") # Título del gráfico
plt.xticks(rotation=90)
plt.show()


# In[22]:


# Contar el número de películas por género
genre_counts = datosp['Genere'].value_counts()
total_peliculas = datosp.shape[0]

# Crear gráfico de pastel
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%')
plt.title("Proporción de Películas por Género en un Año Específico")
plt.legend(genre_counts.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Imprimir nombres de géneros y porcentajes fuera del gráfico
for genre, count in genre_counts.items():
    porcentaje = count/total_peliculas * 100
    print(f"{genre}: {porcentaje:.2f}%")
    
plt.show()


# In[23]:


# Agrupar los datos por género y año
data_grouped = datosp.groupby(['Any', 'Genere']).size().reset_index(name='counts')

# Utilizar la función "unstack()" para mover los géneros a los índices
data_grouped = data_grouped.pivot(index='Any', columns='Genere', values='counts')

# Crear gráfico de línea para cada género
data_grouped.plot(kind='line')

# Añadir título y etiquetas a los ejes
plt.title("Cambio en la cantidad de películas por género a lo largo del tiempo")

