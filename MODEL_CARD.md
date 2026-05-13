# Model Card

## Model Name: C20739_Caleb_Arias

Clasificador Binario de Imágenes v1.0

### Mejor modelo seleccionado

* Modelo: SVM
* Accuracy (Validación Cruzada): 0.8345
* Parámetros: `{'C': 10, 'kernel': 'rbf'}`

## Intended Use

Este modelo fue diseñado para realizar clasificación binaria de imágenes entre las clases:

* `cont_positiva`
* `cont_negativa`

El sistema está orientado a fines académicos y experimentales.

### Out of Scope

* Clasificación de imágenes fuera de las dos clases entrenadas.
* Uso con imágenes altamente degradadas o con distribuciones distintas al dataset original.

## Data Summary

Las imágenes fueron recolectadas manualmente y organizadas en dos carpetas correspondientes a cada clase.

Durante el preprocesamiento:

* Las imágenes fueron convertidas a escala de grises.
* Se aplicó ajuste de brillo dinámico.
* Se aumentó el contraste.
* Todas las imágenes fueron redimensionadas a 128x128 píxeles.
* Las imágenes fueron binarizadas utilizando un umbral fijo.

El dataset puede contener variaciones de:

* Iluminación
* Resolución
* Orientación

## Labeling Process

El etiquetado fue realizado manualmente mediante la organización de imágenes en carpetas separadas por clase.

No se utilizó una herramienta automática de anotación.

La calidad y consistencia de las etiquetas depende de la correcta clasificación manual realizada durante la construcción del dataset.

## Metrics

Se evaluaron múltiples modelos de machine learning utilizando validación cruzada de 5 folds (`cv=5`), esto significa que se se entrena el modelo con cuatro partes y posteriormente se prueba con una, utilizando el conjunto designado para el entrenamiento, este proceso se realiza 5 veces utilizando diferentes divisiones del conjunto de entrenamiento.

Mejor resultado obtenido:

* Modelo: SVM
* Accuracy CV: 0.8345
* Kernel: `rbf`
* Parámetro `C`: 10

Modelos evaluados:

* Decision Tree
* Naive Bayes
* KNN
* SVM
* Random Forest

Métrica principal utilizada:

* Accuracy

El conjunto de prueba fue separado antes del entrenamiento y utilizado únicamente para inferencia final.

## Ethical and Safety Notes

El modelo puede presentar sesgos debido a:

* Condiciones de iluminación específicas.
* Tipos de cámaras utilizadas.
* Distribución limitada del dataset.

Los resultados pueden variar significativamente en escenarios distintos a los utilizados durante el entrenamiento.

## Limitations

El desempeño puede degradarse en casos como:

* Objetos pequeños.
* Oclusiones parciales.
* Imágenes borrosas.
* Ruido excesivo.
* Cambios fuertes de iluminación.
* Imágenes fuera de distribución.

Además, la binarización puede eliminar detalles relevantes presentes en la imagen original.

## Reproducibility

### Requisitos

* Python 3
* pip
* make

### Instalación

```bash
pip install pillow numpy pandas scikit-learn joblib
```

### Entrenamiento completo

```bash
make full_training
```

El entrenamiento ejecuta:

1. Preprocesamiento de imágenes.
2. Vectorización.
3. Generación de CSVs.
4. Separación del conjunto de prueba.
5. Búsqueda de hiperparámetros con `GridSearchCV`.
7. Selección automática del mejor modelo.

### Inferencia

```bash
make test_models
```

### Carga manual del modelo

```python
from joblib import load

# Cargar modelo y scaler
modelo = load('mejor_modelo_obtenido/C20739_Caleb_Arias.joblib')
scaler = load('mejor_modelo_obtenido/scaler.joblib')

# Escalar datos antes de predecir
X_new_scaled = scaler.transform(X_new)
predicciones = modelo.predict(X_new_scaled)
```

### Hardware utilizado

El proyecto fue desarrollado para ejecutarse en CPU y no requiere GPU dedicada.

No se utilizaron aceleradores GPU ni hardware especializado.

### Configuración reproducible

El siguiente fragmento de código muestra la configuración específica para el modelo de SVM que entregó los mejores resultados para la clasificación.

```python
model_config = {
    #...
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear']}
    },
    #...
}

for name, config in model_config.items():
    if config['params']:  # Si hay parámetros para optimizar
        grid = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=5,  # 5-fold cross-validation
            n_jobs=-1, 
            scoring='accuracy',
            verbose=1
        )
        grid.fit(X_train_scaled, y_train)
        
        mejor_modelo = grid.best_estimator_
        cv_accuracy = grid.best_score_
        mejores_params = grid.best_params_
    
    # Guardar el mejor modelo
    file_path = os.path.join(FOLDER_NAME, f"Mejor_{name}.joblib")
    dump(mejor_modelo, file_path)
    print(f"Modelo guardado en: {file_path}")
```

### Archivos generados

* `mejor_modelo_obtenido/C20739_Caleb_Arias.joblib`
* `mejor_modelo_obtenido/scaler.joblib`
