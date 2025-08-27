# Laboratorio 5: Clasificación de Tweets usando Minería de Texto
Emilio Reyes, Silvia Illescas y Michelle Mejia

Este proyecto corresponde al Laboratorio 5 del curso **CC3066 – Data Science**, de la **Universidad del Valle de Guatemala**, en el semestre II – 2025. El objetivo del laboratorio es clasificar tweets sobre desastres naturales utilizando técnicas de minería de texto y aprendizaje automático.

## Archivos del Proyecto

Este repositorio contiene los siguientes archivos:

```

C:.
Lab5.ipynb               # Notebook de Python con el análisis y código.
README.md                # Este archivo README.
sample\_submission.csv     # Ejemplo de formato de envío para Kaggle.
test.csv                  # Conjunto de datos de prueba.
train.csv                 # Conjunto de datos de entrenamiento.
wordcloud\_target0.png     # Nube de palabras para tweets no relacionados con desastres (target = 0).
wordcloud\_target1.png     # Nube de palabras para tweets relacionados con desastres (target = 1).

````

## Descripción del Proyecto

Este proyecto se enfoca en la clasificación de tweets relacionados con desastres naturales. El conjunto de datos proviene del [dataset de Kaggle "Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/c/nlp-getting-started/data), y contiene dos categorías de tweets:

- **target = 1**: Tweets relacionados con desastres.
- **target = 0**: Tweets no relacionados con desastres.

El objetivo final es desarrollar un modelo que pueda predecir si un tweet hace referencia a un desastre natural.

## Pasos Realizados

1. **Carga y Preprocesamiento de Datos**
   - Se cargaron los archivos `train.csv` y `test.csv` en el entorno de Python.
   - Se realizaron tareas de preprocesamiento, como la limpieza del texto:
     - Conversión de texto a minúsculas.
     - Eliminación de caracteres especiales, URLs y emoticones.
     - Eliminación de signos de puntuación y stopwords.
     - Mantenimiento de números relevantes (como el "911").

2. **Análisis Exploratorio de Datos**
   - Se exploraron las palabras más frecuentes en los tweets de cada categoría (desastre y no desastre).
   - Se generaron nubes de palabras visualizando las palabras más comunes en ambos conjuntos de tweets.
     - **wordcloud_target0.png**: Nube de palabras de tweets no relacionados con desastres.
     - **wordcloud_target1.png**: Nube de palabras de tweets relacionados con desastres.

3. **Generación de N-Gramas**
   - Se generaron unigramas, bigramas y trigramas para analizar los patrones de texto y mejorar la clasificación.
   - Se calcularon las frecuencias de los n-gramas y se analizaron para observar qué patrones podían predecir mejor si un tweet era de desastre o no.

4. **Entrenamiento del Modelo**
   - Se entrenaron varios modelos de clasificación, como **Naive Bayes** y **Support Vector Machines (SVM)**, utilizando los datos preprocesados.
   - El modelo fue evaluado utilizando métricas de precisión, recall y F1.

5. **Función de Clasificación**
   - Se desarrolló una función en Python que permite ingresar un nuevo tweet y predecir si está relacionado con un desastre (target = 1) o no (target = 0).

## Uso del Proyecto

### Requisitos
- Python 3.x
- Librerías necesarias:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `nltk`
  - `wordcloud`

### Ejecutar el Notebook
1. Clona o descarga este repositorio.
2. Abre el archivo `Lab5.ipynb` en un entorno de Python (como Jupyter Notebook).
3. Ejecuta las celdas del notebook en orden para ver el análisis exploratorio, la limpieza de datos, y la predicción del modelo.

### Función de Clasificación
Para clasificar un tweet nuevo, utiliza la función `classify_tweet()` en el notebook.

```python
def classify_tweet(tweet):
    # Código para clasificar el tweet
    return prediction  # 0 o 1 (desastre o no desastre)
````

## Resultados

* Se entrenó el modelo con el conjunto de datos de entrenamiento (`train.csv`) y se probó en el conjunto de datos de prueba (`test.csv`).
* La clasificación se realizó con éxito y se obtuvo una alta precisión en la predicción de tweets relacionados con desastres.

## Contribuciones

Este proyecto fue realizado por el grupo de estudiantes del curso **CC3066 – Data Science** de la Universidad del Valle de Guatemala.

## Referencias

* Jurafsky, D., & Martin, J. H. (2014). *Speech and Language Processing*.
* Feinerer, I., Hornik, K., & Meyer, D. (2008). Text Mining Infrastructure in R. *Journal of Statistical Software*, 25(5), 1-54.
* NLTK Documentation: [https://www.nltk.org/](https://www.nltk.org/)
* Wordcloud Documentation: [http://amueller.github.io/word\_cloud/](http://amueller.github.io/word_cloud/)

