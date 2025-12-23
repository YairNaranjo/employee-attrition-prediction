"""
Análisis de Fuga de Talento (Employee Attrition)
Dataset: IBM HR Analytics
Nivel: Junior Data Scientist

Este script realiza un análisis predictivo básico para identificar
empleados con mayor probabilidad de abandonar la empresa.
"""

# ============================================
# IMPORTS
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualizaciones
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("="*60)
print("ANÁLISIS DE FUGA DE TALENTO - IBM HR ANALYTICS")
print("="*60)
print()

# ============================================
# 1. CARGA DE DATOS
# ============================================
print("1. CARGANDO DATOS...")
# Leer el archivo CSV desde la misma carpeta
df = pd.read_csv('datos.csv')

# Convertir Attrition de Yes/No a 1/0
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

print(f"✓ Datos cargados: {df.shape[0]} empleados, {df.shape[1]} variables")
print()

# ============================================
# 2. EXPLORACIÓN BÁSICA
# ============================================
print("2. EXPLORACIÓN BÁSICA DE DATOS")
print("-" * 60)

# Primeras filas
print("\nPrimeras 5 filas:")
print(df.head())
print()

# Información general
print("\nInformación del dataset:")
print(df.info())
print()

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())
print()

# Valores nulos
print("\nValores nulos por columna:")
nulos = df.isnull().sum()
if nulos.sum() == 0:
    print("✓ No hay valores nulos en el dataset")
else:
    print(nulos[nulos > 0])
print()

# Distribución de Attrition
print("\nDistribución de Attrition:")
print(df['Attrition'].value_counts())
print(f"\nPorcentaje de empleados que se fueron: {df['Attrition'].mean()*100:.1f}%")
print()

# ============================================
# 3. VISUALIZACIONES
# ============================================
print("3. GENERANDO VISUALIZACIONES...")

# 3.1 Distribución de Attrition
plt.figure(figsize=(8, 6))
attrition_counts = df['Attrition'].value_counts()
plt.bar(['No se fue (0)', 'Se fue (1)'], attrition_counts.values, 
        color=['green', 'red'], alpha=0.7)
plt.title('Distribución de Attrition', fontsize=14, fontweight='bold')
plt.ylabel('Número de empleados')
plt.xlabel('Attrition')
for i, v in enumerate(attrition_counts.values):
    plt.text(i, v + 10, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('distribucion_attrition.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfico guardado: distribucion_attrition.png")

# 3.2 Salario vs Attrition
plt.figure(figsize=(10, 6))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, palette=['green', 'red'])
plt.title('Salario Mensual vs Attrition', fontsize=14, fontweight='bold')
plt.xlabel('Attrition (0=No, 1=Sí)')
plt.ylabel('Salario Mensual')
plt.tight_layout()
plt.savefig('salario_vs_attrition.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfico guardado: salario_vs_attrition.png")

# 3.3 OverTime vs Attrition
plt.figure(figsize=(10, 6))
overtime_attrition = pd.crosstab(df['OverTime'], df['Attrition'], normalize='index') * 100
overtime_attrition.plot(kind='bar', color=['green', 'red'], alpha=0.7)
plt.title('Porcentaje de Attrition por OverTime', fontsize=14, fontweight='bold')
plt.xlabel('OverTime')
plt.ylabel('Porcentaje (%)')
plt.xticks(rotation=0)
plt.legend(['No se fue', 'Se fue'], title='Attrition')
plt.tight_layout()
plt.savefig('overtime_vs_attrition.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfico guardado: overtime_vs_attrition.png")

# 3.4 Distribución de Edad
plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribución de Edad de Empleados', fontsize=14, fontweight='bold')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.axvline(df['Age'].mean(), color='red', linestyle='--', 
            label=f'Media: {df["Age"].mean():.1f} años')
plt.legend()
plt.tight_layout()
plt.savefig('distribucion_edad.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfico guardado: distribucion_edad.png")

# 3.5 Heatmap de correlación
plt.figure(figsize=(10, 8))
# Seleccionar variables numéricas importantes
corr_vars = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome',
             'JobSatisfaction', 'WorkLifeBalance', 'Attrition']
correlation = df[corr_vars].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlación entre Variables Principales', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap_correlacion.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfico guardado: heatmap_correlacion.png")
print()

# ============================================
# 4. PREPROCESAMIENTO
# ============================================
print("4. PREPROCESAMIENTO DE DATOS...")

# 4.1 Seleccionar columnas importantes
columnas_seleccionadas = [
    'Age', 'MonthlyIncome', 'OverTime', 'JobSatisfaction', 
    'WorkLifeBalance', 'YearsAtCompany', 'DistanceFromHome',
    'JobRole', 'Department', 'Attrition'
]
df_modelo = df[columnas_seleccionadas].copy()
print(f"✓ Seleccionadas {len(columnas_seleccionadas)-1} variables para el modelo")

# 4.2 Verificar valores nulos
if df_modelo.isnull().sum().sum() > 0:
    df_modelo = df_modelo.dropna()
    print(f"✓ Eliminadas filas con valores nulos")
else:
    print("✓ No hay valores nulos")

# 4.3 Convertir variables categóricas
# OverTime: Yes/No a 1/0
df_modelo['OverTime'] = df_modelo['OverTime'].map({'Yes': 1, 'No': 0})

# JobRole y Department: usar LabelEncoder
le_job = LabelEncoder()
le_dept = LabelEncoder()
df_modelo['JobRole'] = le_job.fit_transform(df_modelo['JobRole'])
df_modelo['Department'] = le_dept.fit_transform(df_modelo['Department'])
print("✓ Variables categóricas convertidas a numéricas")

# 4.4 Separar variables X (features) y (target)
X = df_modelo.drop('Attrition', axis=1)
y = df_modelo['Attrition']
print(f"✓ Variables separadas: X={X.shape}, y={y.shape}")

# 4.5 Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Datos divididos: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# 4.6 Escalar variables numéricas
scaler = StandardScaler()
# Columnas numéricas para escalar
columnas_numericas = ['Age', 'MonthlyIncome', 'YearsAtCompany', 
                      'DistanceFromHome', 'JobSatisfaction', 'WorkLifeBalance']
X_train[columnas_numericas] = scaler.fit_transform(X_train[columnas_numericas])
X_test[columnas_numericas] = scaler.transform(X_test[columnas_numericas])
print("✓ Variables numéricas escaladas con StandardScaler")
print()

# ============================================
# 5. MODELADO
# ============================================
print("5. ENTRENANDO MODELO...")

# Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)
print("✓ Modelo Random Forest entrenado exitosamente")

# Hacer predicciones
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad de Attrition=1
print("✓ Predicciones realizadas")
print()

# ============================================
# 6. EVALUACIÓN DEL MODELO
# ============================================
print("6. EVALUACIÓN DEL MODELO")
print("-" * 60)

# 6.1 Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy*100:.2f}%")
print("(Porcentaje de predicciones correctas)")

# 6.2 Classification Report
print("\nClassification Report:")
print("(Precision: de los que predije positivos, cuántos acerté)")
print("(Recall: de los positivos reales, cuántos identifiqué)")
print("(F1-score: balance entre precision y recall)")
print()
print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition']))

# 6.3 Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfico guardado: matriz_confusion.png")

# 6.4 Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('curva_roc.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Gráfico guardado: curva_roc.png (AUC = {roc_auc:.2f})")
print()

# ============================================
# 7. FEATURE IMPORTANCE
# ============================================
print("7. IMPORTANCIA DE VARIABLES")
print("-" * 60)

# Obtener importancia de features
feature_importance = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': model.feature_importances_
}).sort_values('Importancia', ascending=False)

print("\nTop 10 variables más importantes:")
print(feature_importance.head(10))
print()

# Gráfico de Feature Importance
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(top_features['Variable'], top_features['Importancia'], color='steelblue')
plt.xlabel('Importancia')
plt.title('Top 10 Variables Más Importantes', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfico guardado: feature_importance.png")

# Interpretación simple
var_mas_importante = feature_importance.iloc[0]['Variable']
print(f"\nInterpretación: La variable '{var_mas_importante}' es la más importante")
print("para predecir si un empleado se irá o no. Esto significa que esta")
print("característica tiene el mayor poder predictivo en nuestro modelo.")
print()

# ============================================
# 8. GUARDAR RESULTADOS
# ============================================
print("8. GUARDANDO RESULTADOS...")

# Preparar DataFrame con resultados
# Obtener datos originales del test set
indices_test = X_test.index
df_resultados = df.loc[indices_test].copy()

# Agregar predicciones y probabilidades
df_resultados['Prediccion'] = y_pred
df_resultados['Probabilidad_Attrition'] = y_pred_proba

# Ordenar por probabilidad (mayor riesgo primero)
df_resultados = df_resultados.sort_values('Probabilidad_Attrition', ascending=False)

# Guardar a CSV
df_resultados.to_csv('resultados_predicciones.csv', index=False)
print(f"✓ Resultados guardados en: resultados_predicciones.csv")
print(f"  Total de registros: {len(df_resultados)}")
print()

# Mostrar algunos ejemplos de alto riesgo
print("Empleados con mayor probabilidad de Attrition:")
print(df_resultados[['Age', 'MonthlyIncome', 'OverTime', 'JobSatisfaction', 
                     'Probabilidad_Attrition']].head(5))
print()

# ============================================
# 9. CONCLUSIONES
# ============================================
print("="*60)
print("CONCLUSIONES PRINCIPALES")
print("="*60)
print()

empleados_alto_riesgo = len(df_resultados[df_resultados['Probabilidad_Attrition'] > 0.5])
tasa_attrition = df['Attrition'].mean() * 100

print(f"• El modelo alcanzó un accuracy de {accuracy*100:.1f}% en el conjunto de test")
print(f"  (esto significa que acierta en {accuracy*100:.1f} de cada 100 predicciones)")
print()

print(f"• La variable más importante para predecir attrition es: {var_mas_importante}")
print(f"  seguida de las otras variables en el gráfico de importancia")
print()

print(f"• Identificamos {empleados_alto_riesgo} empleados con alta probabilidad de irse")
print(f"  (probabilidad > 50%) en el conjunto de test")
print()

print(f"• La tasa de attrition en la empresa es de {tasa_attrition:.1f}%")
print(f"  (aproximadamente {int(df['Attrition'].sum())} de {len(df)} empleados)")
print()

print("• LIMITACIONES: Este es un modelo básico que puede mejorarse con:")
print("  - Feature engineering (crear nuevas variables)")
print("  - Prueba de múltiples modelos (XGBoost, Logistic Regression, etc.)")
print("  - Tuning de hiperparámetros")
print("  - Cross-validation para mejor validación")
print("  - Análisis más profundo de casos mal clasificados")
print()

print("="*60)
print("ANÁLISIS COMPLETADO EXITOSAMENTE")
print("="*60)
print()
print("Archivos generados:")
print("  1. distribucion_attrition.png")
print("  2. salario_vs_attrition.png")
print("  3. overtime_vs_attrition.png")
print("  4. distribucion_edad.png")
print("  5. heatmap_correlacion.png")
print("  6. matriz_confusion.png")
print("  7. curva_roc.png")
print("  8. feature_importance.png")
print("  9. resultados_predicciones.csv")
