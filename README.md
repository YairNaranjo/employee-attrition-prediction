#  Análisis Predictivo de Fuga de Talento (Employee Attrition)

## Descripción del Proyecto

Este proyecto analiza datos de recursos humanos para predecir qué empleados tienen mayor probabilidad de abandonar la empresa (attrition). Utilizando el dataset de IBM HR Analytics, se desarrolló un modelo de Machine Learning que identifica patrones y factores de riesgo asociados a la rotación de personal.

El objetivo principal es demostrar competencias en análisis de datos y machine learning a nivel junior, implementando un flujo completo desde la exploración de datos hasta la generación de predicciones accionables para el área de Recursos Humanos.

Este análisis incluye exploración de datos, visualizaciones, preprocesamiento, modelado con Random Forest y evaluación de resultados, todo implementado con código limpio y bien documentado.

##  Resultados Principales

- **Accuracy del modelo**: ~85% (varía según la ejecución por aleatoriedad)
- **Variables más importantes**: OverTime, MonthlyIncome, Age, YearsAtCompany
- **Tasa de attrition**: ~16% de los empleados
- **Empleados de alto riesgo identificados**: Se generan predicciones individuales con probabilidades

##  Requisitos

### Librerías de Python necesarias:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### Instalación:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

##  Cómo Ejecutar el Proyecto

### Paso 1: Preparar archivos
Asegúrate de tener estos dos archivos en la misma carpeta:
- `datos.csv` (dataset de IBM HR Analytics)
- `analisis_attrition_junior.py` (script principal)

### Paso 2: Ejecutar el script
```bash
python analisis_attrition_junior.py
```

### Paso 3: Revisar resultados
El script generará automáticamente:
- 8 gráficos en formato PNG
- 1 archivo CSV con predicciones
- Métricas de evaluación en consola

**Tiempo de ejecución aproximado**: 10-15 segundos

##  Archivos Generados

### Visualizaciones (PNG):
1. **distribucion_attrition.png** - Cantidad de empleados que se fueron vs se quedaron
2. **salario_vs_attrition.png** - Comparación de salarios entre grupos
3. **overtime_vs_attrition.png** - Impacto de horas extra en attrition
4. **distribucion_edad.png** - Distribución de edades en la empresa
5. **heatmap_correlacion.png** - Correlación entre variables principales
6. **matriz_confusion.png** - Rendimiento del modelo (VP, VN, FP, FN)
7. **curva_roc.png** - Capacidad discriminativa del modelo (AUC)
8. **feature_importance.png** - Variables más importantes del modelo

### Datos:
- **resultados_predicciones.csv** - Contiene:
  - Todas las columnas originales del dataset
  - Columna `Prediccion`: 0 (no se irá) o 1 (se irá)
  - Columna `Probabilidad_Attrition`: probabilidad de 0 a 1
  - Ordenado por probabilidad descendente (mayor riesgo primero)

##  Estructura del Análisis

### 1. Carga y Exploración
- Lectura del CSV
- Conversión de Attrition (Yes/No → 1/0)
- Análisis descriptivo con `head()`, `info()`, `describe()`
- Verificación de valores nulos
- Análisis de balance de clases

### 2. Visualizaciones
- 5 gráficos exploratorios para entender los datos
- Uso de matplotlib y seaborn
- Gráficos limpios y profesionales

### 3. Preprocesamiento
- Selección de 9 variables clave (de 35 originales)
- Conversión de variables categóricas con LabelEncoder
- División train/test (80/20) con estratificación
- Escalado de variables numéricas con StandardScaler

### 4. Modelado
- Random Forest Classifier
- 100 árboles de decisión
- Parámetros por defecto (enfoque junior)
- Predicciones con probabilidades

### 5. Evaluación
- **Accuracy**: porcentaje de aciertos totales
- **Precision**: de los que predije que se van, cuántos realmente se van
- **Recall**: de los que se van, cuántos logré identificar
- **F1-score**: balance entre precision y recall
- **Matriz de confusión**: detalle de aciertos y errores
- **Curva ROC y AUC**: capacidad discriminativa del modelo

### 6. Interpretación
- Feature Importance: qué variables son más importantes
- Análisis de empleados de alto riesgo
- Insights accionables para RH

##  Recomendaciones de Negocio

Basado en los resultados del modelo:

1. **Monitoreo de OverTime**: Los empleados que trabajan horas extra tienen significativamente mayor probabilidad de irse. Considerar políticas de balance vida-trabajo.

2. **Revisión Salarial**: El salario mensual es un factor predictivo importante. Realizar benchmarking salarial periódico y ajustes competitivos.

3. **Retención Temprana**: Empleados con menos años en la compañía tienen mayor riesgo. Implementar programa de onboarding robusto y seguimiento en primeros 2 años.

4. **Intervención Proactiva**: Usar las predicciones del modelo para identificar empleados de alto riesgo y tomar acciones preventivas (conversaciones 1-on-1, planes de carrera, ajustes de compensación).

5. **Satisfacción Laboral**: Mantener encuestas regulares de satisfacción laboral y work-life balance, ya que son factores predictivos importantes.

## 📊 Variables Utilizadas

| Variable | Descripción |
|----------|-------------|
| Age | Edad del empleado |
| MonthlyIncome | Salario mensual |
| OverTime | Si trabaja horas extra (Yes/No) |
| JobSatisfaction | Satisfacción laboral (1-4) |
| WorkLifeBalance | Balance vida-trabajo (1-4) |
| YearsAtCompany | Años en la empresa |
| DistanceFromHome | Distancia casa-trabajo (km) |
| JobRole | Rol/puesto del empleado |
| Department | Departamento |

##  Conceptos Aplicados

- **Análisis Exploratorio de Datos (EDA)**
- **Limpieza y preprocesamiento de datos**
- **Encoding de variables categóricas**
- **Feature scaling (estandarización)**
- **Train-Test split**
- **Clasificación binaria**
- **Random Forest**
- **Métricas de evaluación**
- **Feature importance**
- **Visualización de datos**

##  Limitaciones y Mejoras Futuras

### Limitaciones del modelo actual:
- No se realizó feature engineering avanzado
- Solo se probó un tipo de modelo (Random Forest)
- No se aplicó cross-validation
- No se optimizaron hiperparámetros
- Podría existir desbalance de clases no tratado

### Mejoras posibles:
- Probar otros modelos (XGBoost, Logistic Regression, SVM)
- Implementar Grid Search para optimizar hiperparámetros
- Aplicar técnicas de balanceo (SMOTE, undersampling)
- Crear nuevas variables (feature engineering)
- Usar K-Fold Cross-Validation
- Analizar casos mal clasificados en detalle
- Implementar un pipeline automatizado

##  Notas Técnicas

- El modelo usa `random_state=42` para reproducibilidad
- Las visualizaciones se guardan en alta resolución (300 DPI)
- El código está diseñado para ser legible y educativo
- Se priorizó claridad sobre optimización de performance
- Total de líneas de código: ~300

##  Uso y Licencia

Este proyecto es de código abierto con fines educativos y de portafolio. Siéntete libre de usar, modificar y compartir con atribución apropiada.

---

**Dataset**: IBM HR Analytics Employee Attrition Dataset  
**Lenguaje**: Python 3.7+  
