import os
import pandas as pd
import numpy as np
from joblib import load

# Rutas
testing_negative_path = 'testing_set/negative_test.csv'
testing_positive_path = 'testing_set/positive_test.csv'
modelo_path = 'mejor_modelo_obtenido/C20739_Caleb_Arias.joblib'
scaler_path = 'mejor_modelo_obtenido/scaler.joblib'

# Cargar modelo y scaler
print("Cargando modelo y scaler...")
modelo = load(modelo_path)
scaler = load(scaler_path)

# Cargar datos
print("Cargando datos de prueba...")
df_neg = pd.read_csv(testing_negative_path, header=None)
df_pos = pd.read_csv(testing_positive_path, header=None)

# Combinar datos
X_test = np.vstack((df_neg.values, df_pos.values))

# Escalar los datos antes de predecir
print("Escalando datos...")
X_test_scaled = scaler.transform(X_test)

# Predicciones
print("Realizando predicciones...")
y_pred = modelo.predict(X_test_scaled)

# Crear etiquetas reales
y_true = np.array([0]*len(df_neg) + [1]*len(df_pos))

# Crear CSV con resultados
resultados = pd.DataFrame({
    'clase_esperada': y_true,
    'clase_predicha': y_pred,
    'clase_esperada_nombre': ['negativo' if y == 0 else 'positivo' for y in y_true],
    'clase_predicha_nombre': ['negativo' if y == 0 else 'positivo' for y in y_pred]
})

# Guardar CSV
output_path = 'inferencia.csv'
resultados.to_csv(output_path, index=False)
print(f"\nCSV guardado en: {output_path}")

print(f"\nResumen:")
print(f"Total muestras: {len(resultados)}")
print(f"Negativos reales: {sum(y_true == 0)}")
print(f"Positivos reales: {sum(y_true == 1)}")
print(f"Predichos como negativos: {sum(y_pred == 0)}")
print(f"Predichos como positivos: {sum(y_pred == 1)}")

# Calcular accuracy
accuracy = np.mean(y_true == y_pred)
print(f"\nAccuracy en conjunto de prueba: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Calcular métricas de error
falsos_positivos = np.sum((y_true == 0) & (y_pred == 1))
falsos_negativos = np.sum((y_true == 1) & (y_pred == 0))
verdaderos_positivos = np.sum((y_true == 1) & (y_pred == 1))
verdaderos_negativos = np.sum((y_true == 0) & (y_pred == 0))

print(f"\nMétricas de clasificación:")
print(f"={'='*50}")
print(f"Verdaderos Positivos (VP): {verdaderos_positivos}")
print(f"Verdaderos Negativos (VN): {verdaderos_negativos}")
print(f"Falsos Positivos (FP):     {falsos_positivos} ← predijo positivo, era negativo")
print(f"Falsos Negativos (FN):     {falsos_negativos} ← predijo negativo, era positivo")
print(f"={'='*50}")
