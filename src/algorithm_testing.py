import os
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Modelos
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# --- RUTAS DE LOS DATASETS ---
training_negative_path = 'training_set/negative_training.csv'
training_positive_path = 'training_set/positive_training.csv'
testing_negative_path  = 'testing_set/negative_test.csv'
testing_positive_path  = 'testing_set/positive_test.csv'

FOLDER_NAME = "models"
if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)

# --- FUNCIÓN PARA CARGAR DATOS ---
def cargar_datos_desde_archivos(path_neg, path_pos):
    # Carga de archivos completos
    df_neg = pd.read_csv(path_neg, header=None)
    df_pos = pd.read_csv(path_pos, header=None)
    
    X_n = df_neg.values
    X_p = df_pos.values
    
    # Creación de etiquetas (0 para negativo, 1 para positivo)
    y_n = np.zeros(X_n.shape[0])
    y_p = np.ones(X_p.shape[0])
    
    X_total = np.vstack((X_n, X_p))
    y_total = np.concatenate((y_n, y_p))
    
    return X_total, y_total

# --- 1. CARGA DE DATOS ---
print("Cargando datos de Entrenamiento...")
X_train, y_train = cargar_datos_desde_archivos(training_negative_path, training_positive_path)

print("Cargando datos de Prueba...")
X_test, y_test = cargar_datos_desde_archivos(testing_negative_path, testing_positive_path)

# --- 2. CONFIGURACIÓN DINÁMICA ---
N_COLUMNAS = X_train.shape[1]
print(f"Detectadas {N_COLUMNAS} columnas por vector.")

# --- 3. PREPROCESAMIENTO ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. DICCIONARIO DE MODELOS ---
model_config = {
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {'max_depth': [5, 10, 15, 20], 'criterion': ['gini', 'entropy']}
    },
    'NaiveBayes': {
        'model': GaussianNB(),
        'params': {} 
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors': [3, 5, 7, 9, 11, 13, 15], 'weights': ['distance']}
    },
    'SVM': {
        'model': SVC(probability=True),
        'params': {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear']}
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [50,100,200,500], 'max_depth': [None, 10]}
    }
}

# --- 5. ENTRENAMIENTO Y EVALUACIÓN ---
resultados = []

for name, config in model_config.items():
    print(f"\nOptimizando {name}...")
    
    grid = GridSearchCV(config['model'], config['params'], cv=3, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train_scaled, y_train)
    
    mejor_modelo = grid.best_estimator_
    
    y_pred = mejor_modelo.predict(X_test_scaled)
    acc_test = accuracy_score(y_test, y_pred)
    
    file_path = os.path.join(FOLDER_NAME, f"Mejor_{name}.joblib")
    dump(mejor_modelo, file_path)
    
    resultados.append({
        'Modelo': name, 
        'Mejores_Params': grid.best_params_,
        'CV_Accuracy': f"{grid.best_score_:.4f}", 
        'Test_Accuracy': f"{acc_test:.4f}"
    })
    
    print(f"Guardado: {file_path}")
    print(f"Accuracy en Prueba: {acc_test:.4f}")

# --- 6. RESUMEN FINAL ---
print("\n" + "="*50)
print("RESUMEN DE EVALUACIÓN FINAL")
print("="*50)
df_resumen = pd.DataFrame(resultados)
print(df_resumen[['Modelo', 'CV_Accuracy', 'Test_Accuracy']])