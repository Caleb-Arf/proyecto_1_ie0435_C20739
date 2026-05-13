# Dataset

## Recolección de datos

El dataset fue construido a partir de imágenes organizadas manualmente en dos clases:

* `cont_positiva`
* `cont_negativa`

Las imágenes originales fueron almacenadas en:

```text
sample_images/
├── cont_positiva/
└── cont_negativa/
```

Estas imágenes deben ser procesadas debido a que estas no se tomaron en ambientes estrictamente controlados y carecen de estandarización, esta falta de estandarización hace que unas sean muy distintas de otras en cuando a condiciones como iluminación y contraste, para realizar un preprocesamiento adecuado de las mismas se aplica un filtro en escala de grises para identificar los contornos de los objetos, posteriormente se realizó un ajuste de brillo dinámico a cada imágen hasta llevar la imagen a un brillo promedio predefinido, de este modo se puede minimizar el efecto de los ambientes poco controlados, posteriormente se realizó un aumento en el contraste de cada imágen para divisar mejor las fronteras de los objetos a identificar.

Finalmente se redimensionó la imágen a tamaño 128x128 y utilizando un umbral para cada pixel se vectorizó la imágen, el umbral utilizado tiene un valor de 128 por que el valor mínimo es 0 y el máximo 255, se utilizó la mitad tomando en cuenta que el valor promedio de brillo en las imágenes es 128 así se pueden distinguir los pixeles de mejor manera. Los vectores generados fueron almacenados en archivos CSV para entrenamiento y posteriormente se separó del conjunto de entrenamiento los datos de prueba.

## Limitaciones

* El dataset depende de la calidad y variedad de las imágenes originales.
* Transformaciones como escala de grises y redimensionamiento generan pérdida de información. 
* La binarización y la mala Iluminación eliminan detalles relevantes presentes en intensidades intermedias.
* Un dataset pequeño o poco diverso puede provocar sobreajuste.
* El modelo puede presentar bajo desempeño frente a imágenes con iluminación, orientación o ruido distintos a los observados durante el entrenamiento.
* La separación automática del conjunto de prueba no es aleatoria completa, por lo que podría introducir sesgos en la evaluación.
