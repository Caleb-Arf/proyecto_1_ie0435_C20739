import os
import shutil
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
# Modelos
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Datasets (solo entrenamiento)
training_negative_path = 'training_set/negative_training.csv'
training_positive_path = 'training_set/positive_training.csv'

FOLDER_NAME = "models"
BEST_MODEL_FOLDER = "mejor_modelo_obtenido"

if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)
if not os.path.exists(BEST_MODEL_FOLDER):
    os.makedirs(BEST_MODEL_FOLDER)

# Función para cargar datos desde csvs
def cargar_datos_desde_archivos(path_neg, path_pos):
    df_neg = pd.read_csv(path_neg, header=None)
    df_pos = pd.read_csv(path_pos, header=None)
    
    X_n = df_neg.values
    X_p = df_pos.values
    
    y_n = np.zeros(X_n.shape[0])
    y_p = np.ones(X_p.shape[0])
    
    X_total = np.vstack((X_n, X_p))
    y_total = np.concatenate((y_n, y_p))
    
    return X_total, y_total

# Cargar datos de entrenamiento
print("Cargando datos de Entrenamiento...")
X_train, y_train = cargar_datos_desde_archivos(training_negative_path, training_positive_path)

N_COLUMNAS = X_train.shape[1]
print(f"Detectadas {N_COLUMNAS} columnas por vector.")
print(f"Total de muestras: {X_train.shape[0]} ({np.sum(y_train == 0):.0f} negativas, {np.sum(y_train == 1):.0f} positivas)")

# Escalado de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Guardar el scaler para uso posterior en inferencia
scaler_path = os.path.join(FOLDER_NAME, "scaler.joblib")
dump(scaler, scaler_path)
print(f"\nScaler guardado en: {scaler_path}")

# Diccionario para listar los modelos y parámetros
model_config = {
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {'max_depth': [5, 10, 15, 20], 'criterion': ['gini', 'entropy']}
    },
    'NaiveBayes': {
        'model': GaussianNB(),
        'params': {}  # Naive Bayes no tiene hiperparámetros importantes para ajustar
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors': [3, 5, 7, 9, 11, 13, 15], 'weights': ['uniform', 'distance']}
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear']}
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [50, 100, 200, 500], 'max_depth': [None, 10, 20]}
    }
}

# Entrenamiento de modelos con validación cruzada
resultados = []

for name, config in model_config.items():
    print(f"\n{'='*60}")
    print(f"Optimizando {name}...")
    print('='*60)
    
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
        
        print(f"\nMejores parámetros encontrados: {mejores_params}")
        print(f"Accuracy en Validación Cruzada: {cv_accuracy:.4f}")
        
        cv_std = "N/A"
        
    else:  # Para modelos sin hiperparámetros (ej. Naive Bayes)
        mejor_modelo = config['model']
        mejor_modelo.fit(X_train_scaled, y_train)
        
        # Calcular accuracy con validación cruzada
        cv_scores = cross_val_score(mejor_modelo, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        cv_accuracy = cv_scores.mean()
        mejores_params = "N/A (sin hiperparámetros)"
        cv_std = cv_scores.std()
        
        print(f"\nAccuracy en Validación Cruzada: {cv_accuracy:.4f} (+/- {cv_std * 2:.4f})")
    
    # Guardar el mejor modelo
    file_path = os.path.join(FOLDER_NAME, f"Mejor_{name}.joblib")
    dump(mejor_modelo, file_path)
    print(f"Modelo guardado en: {file_path}")
    
    # Almacenar resultados
    resultados.append({
        'Modelo': name, 
        'Mejores_Parametros': str(mejores_params),
        'CV_Accuracy': cv_accuracy,  # Mantener como float para ordenar
        'CV_Accuracy_str': f"{cv_accuracy:.4f}",
        'CV_Std': f"{cv_std:.4f}" if cv_std != "N/A" else "N/A",
        'file_path': file_path
    })

# Guardar y mostrar resultados
print("\n" + "="*80)
print("RESUMEN DE EVALUACIÓN FINAL - VALIDACIÓN CRUZADA")
print("="*80)

df_resumen = pd.DataFrame(resultados)
df_resumen = df_resumen.sort_values('CV_Accuracy', ascending=False).reset_index(drop=True)

# Mostrar en terminal (sin la columna file_path)
print(df_resumen[['Modelo', 'CV_Accuracy_str', 'Mejores_Parametros', 'CV_Std']].to_string(index=False))

# Guardar en archivo .txt con formato mejorado
results_file = os.path.join(BEST_MODEL_FOLDER, 'resultados_entrenamiento.txt')
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RESUMEN DE EVALUACIÓN FINAL - VALIDACIÓN CRUZADA\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset: {N_COLUMNAS} características\n")
    f.write(f"Muestras de entrenamiento: {X_train.shape[0]}\n")
    f.write(f"  - Negativas: {np.sum(y_train == 0):.0f}\n")
    f.write(f"  - Positivas: {np.sum(y_train == 1):.0f}\n")
    f.write("\n" + "="*80 + "\n\n")
    f.write(df_resumen[['Modelo', 'CV_Accuracy_str', 'Mejores_Parametros', 'CV_Std']].to_string(index=False))
    f.write("\n\n" + "="*80 + "\n")
    f.write("DETALLES POR MODELO:\n")
    f.write("="*80 + "\n\n")
    
    for idx, row in df_resumen.iterrows():
        f.write(f"{idx + 1}. {row['Modelo']}\n")
        f.write(f"   Accuracy CV: {row['CV_Accuracy_str']}\n")
        f.write(f"   Parámetros: {row['Mejores_Parametros']}\n")
        if row['CV_Std'] != "N/A":
            f.write(f"   Desviación estándar: {row['CV_Std']}\n")
        f.write("\n")

print(f"\nResultados guardados en: {results_file}")

# Identificar el mejor modelo
mejor_resultado = df_resumen.iloc[0]
print("\n" + "="*80)
print("MEJOR MODELO IDENTIFICADO:")
print("="*80)
print(f"Modelo: {mejor_resultado['Modelo']}")
print(f"Accuracy (CV): {mejor_resultado['CV_Accuracy_str']}")
print(f"Parámetros: {mejor_resultado['Mejores_Parametros']}")
print("="*80)

# Copiar el mejor modelo y el scaler a la carpeta mejor_modelo_obtenido
print(f"\n{'='*80}")
print("COPIANDO MEJOR MODELO Y SCALER...")
print('='*80)

# Copiar el mejor modelo
mejor_modelo_origen = mejor_resultado['file_path']
mejor_modelo_destino = os.path.join(BEST_MODEL_FOLDER, "C20739_Caleb_Arias.joblib")
shutil.copy2(mejor_modelo_origen, mejor_modelo_destino)
print(f"✓ Mejor modelo copiado a: {mejor_modelo_destino}")

# Copiar el scaler
scaler_destino = os.path.join(BEST_MODEL_FOLDER, "scaler.joblib")
shutil.copy2(scaler_path, scaler_destino)
print(f"✓ Scaler copiado a: {scaler_destino}")

# Crear un archivo de metadatos del mejor modelo
metadata_file = os.path.join(BEST_MODEL_FOLDER, 'mejor_modelo_info.txt')
with open(metadata_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("INFORMACIÓN DEL MEJOR MODELO\n")
    f.write("="*80 + "\n\n")
    f.write(f"Modelo: {mejor_resultado['Modelo']}\n")
    f.write(f"Accuracy (Validación Cruzada): {mejor_resultado['CV_Accuracy_str']}\n")
    if mejor_resultado['CV_Std'] != "N/A":
        f.write(f"Desviación Estándar: {mejor_resultado['CV_Std']}\n")
    f.write(f"Parámetros: {mejor_resultado['Mejores_Parametros']}\n")
    f.write(f"\nArchivo del modelo: C20739_Caleb_Arias.joblib\n") 
    f.write(f"Archivo del scaler: scaler.joblib\n")
    f.write("\n" + "="*80 + "\n")
    f.write("INSTRUCCIONES PARA INFERENCIA:\n")
    f.write("="*80 + "\n\n")
    f.write("Para cargar y usar este modelo:\n\n")
    f.write("from joblib import load\n\n")
    f.write(f"# Cargar modelo y scaler\n")
    f.write(f"modelo = load('mejor_modelo_obtenido/C20739_Caleb_Arias.joblib')\n") 
    f.write(f"scaler = load('mejor_modelo_obtenido/scaler.joblib')\n\n")
    f.write(f"# Escalar datos antes de predecir\n")
    f.write(f"X_new_scaled = scaler.transform(X_new)\n")
    f.write(f"predicciones = modelo.predict(X_new_scaled)\n")

print(f"✓ Información del modelo guardada en: {metadata_file}")

print(f"\n{'='*80}")
print("PROCESO COMPLETADO EXITOSAMENTE")
print('='*80)
print(f"\nArchivos en '{BEST_MODEL_FOLDER}':")
print(f"  1. C20739_Caleb_Arias.joblib - Mejor modelo entrenado")
print(f"  2. scaler.joblib - Scaler para preprocesamiento")
print(f"  3. resultados_entrenamiento.txt - Resumen de todos los modelos")
print(f"  4. mejor_modelo_info.txt - Información y guía de uso del mejor modelo")
print("="*80)