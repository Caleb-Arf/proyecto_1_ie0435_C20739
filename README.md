# Proyecto 1 IE0435: Clasificación de Imágenes

Para este proyecto se realizó un pipeline para preprocesamiento, vectorización, entrenamiento y evaluación de modelos de clasificación binaria utilizando imágenes que se clasifican correspondientemente en contaminación positiva y negativa.

## Requisitos

* Python 3
* pip
* make

## Instalación

Instalar dependencias:

```bash
pip install pillow numpy pandas scikit-learn joblib
```

## Estructura esperada

```text
.
├── Makefile
├── src/
│   ├── image_preprocessor.py
│   ├── vectoring.py
│   ├── generate_csv.py
│   ├── testing_set.py
│   ├── model_training.py
│   └── model_test.py
├── sample_images/
│   ├── cont_positiva/
│   └── cont_negativa/
```

Las imágenes originales provienen del dataset reunido por las fotos tomadas por los integrantes del curso, el dataset utilizado para el entrenamiento de los modelos se encuentran en:

 https://drive.google.com/drive/folders/1z8dIUDDkutQPj2zNj_W4l9mWH3CcBYVJ?usp=sharing.

Esta carpeta debe reemplazar a la carpeta:
* `sample_images`

O bien las imágenes contenidas deben colocarse en:

* `sample_images/cont_positiva`
* `sample_images/cont_negativa`

## Entrenamiento completo

Antes de iniciar con el proceso del pipeline, por favor asegurarse que el dataset tiene imágenes en las carpetas correspondientes a contaminación negativa y contaminación positiva para realizar el entrenamiento de manera correcta.

Ejecuta todo el pipeline:

```bash
make full_training
```

Esto realiza:

1. Preprocesamiento de imágenes
2. Vectorización
3. Generación de CSVs
4. Separación del conjunto de prueba
5. Entrenamiento y selección del mejor modelo

## Inferencia / Evaluación

Ejecutar pruebas sobre el conjunto no visto:

```bash
make test_models
```

El resultado se almacena en:

```text
inferencia.csv
```

## Comandos individuales

Aparte del entrenamiento completo que realiza todo el proceso de preprocesamiento de imágenes, vectorización, generación de archivos y entrenamiento de modelos, el proyecto cuenta con reglas específicas del Makefile para cada proceso individual, de manera que no sea necesario realizar todo el pipeline, si no que se pueden realizar procesos individuales en caso de ser necesario.

Para realizar el preprocesamiento de imágenes que incluye reescalamiento a tamaño 128x128 y aplicar filtro de escala de grises así como homogenizar el brillo de las mismas para poder vectorizarlas se debe correr este comando:

```bash
make image_preprocessing
```

Para convertir las imágenes a vectores se clasifica pixel por pixel basado en el valor de escala de grises utilizando un umbral para decidir si se mapea un 1 o un 0 en cada pixel, posteriormente se presenta la imágen como un vector de 1 fila y 16384 columnas se utiliza el siguiente comando:

```bash
make vectoring
```

Para generar el archivo CSV con todos los datos para el entrenamiento se utiliza el siguiente comando, importante siempre es necesario con cada entrenamiento asegurarse de que este se corra primero que testing_set:

```bash
make trainig_set
```

El siguiente comando se corre después de generar el csv con los datos generales (training_set), se debe correr así debido a que este script separa muestras de prueba eliminandolas del data set de entrenamiento para obtener muestras que el modelo no ha visto, estas se utilizan en la sección de inferencia para poder evaluar la capacidad del modelo:

```bash
make testing_set
```

El siguiente comando corre los entrenamientos de múltiples modelos con múltiples combinaciones de parámetros y de acuerdo al accuracy separa el modelo con mejores resultados para la clasificación:

```bash
make train_models
```

Finalmente, cuando se han entrenado todos los modelos y la carpeta mejor_modelo_obtenido tiene el modelo con los mejore resultados se realiza inferencia para la clasificación y evaluación del desempeño del modelo con datos desconocidos:

```bash
make test_models
```

## Archivos generados importantes

```text
mejor_modelo_obtenido/
├── Mejor_Modelo_*.joblib
├── scaler.joblib
├── resultados_entrenamiento.txt
└── mejor_modelo_info.txt
```

## Limpiar archivos generados

La Regla clean borra todas las carpetas generadas por las reglas anteriores, en caso de borrar todo, asegurarse llenar el conjunto de sample_images con las imágenes correspondientes y correr la regla make full_training para generar todos los archivos y modelos de nuevo.

```bash
make clean
```
