# Proyecto 1 IE0435

## Descripción General

Este proyecto utiliza python para el prepocesamiento de imágenes y vectorización de las mismas para entrenar un modelo de clasificación de imágenes.

El proyecto tiene las siguientes etapas:

1. **Preprocesamiento de imágenes**

   * Conversión a escala de grises
   * Mejora de contraste
   * Redimensionamiento a `128 x 128`

2. **Vectorización**

   * Aplicación de umbral binario
   * Conversión de cada imagen en una matriz binaria

3. **Generación de CSV**

   * Aplanado de matrices binarias en vectores
   * Creación de un dataset estructurado

---

## Requisitos

```bash id="n3u5yt"
pip install pillow numpy
```

---

## Uso con Makefile

### Ejecutar preprocesamiento

```bash id="q8m6lf"
make image_preprocessing
```

### Ejecutar vectorización

```bash id="0f4sjk"
make vectoring
```

### Generar dataset CSV

```bash id="z4p7ac"
make csv
```

### Ejecutar todo el flujo

```bash id="k5u9dw"
make image_preprocessing
make vectoring
make csv
```

### Limpiar archivos generados

```bash id="a9m2ph"
make clean
```

---

## Salidas Generadas

* `resized_images/` → imágenes procesadas
* `image_vectors/` → matrices binarias en texto
* `vectors.csv` → dataset final

---
