# CREDIT1

# Memoria de Machine Learning: Predicción de Clasificación

## 1.Resumen Ejecutivo

El objetivo es evaluar varios modelos de ML, para ver cual es el que mejor predice si un préstamo será pagado o impagado, teniendo en cuenta unas serie de características del usuario-cliente.

## 2.Definición del Problema

- **2.1Descripción del Problema:** Predecir de antemano(antes de hacer la inversión), o sea antes de conceder el préstamo, si será pagado o impagado.

- **2.2Importancia:** Rentabilizar los recursos empleados en conceder financiación a los clientes y que estos recursos no sean generadores de pérdidas, al prestarse a clientes con probabilidad de impago.

## 3.Exploración de Datos

- **3.1.Origen de los Datos:** El dataset que he utilizado para este estudio, esta en el siguiente link, https://www.kaggle.com/datasets/jeandedieunyandwi/lending-club-dataset/data, tiene 396.030 instancias, y 27 características.

- **3.2.Análisis Exploratorio de Datos (EDA):** Los detalles correspondientes a este apartado se encuentran en el siguiente enlace [EDA](../notebooks/002_LimpiezaEDA.ipynb)

## 4.Preprocesamiento de Datos

- **4.1.Limpieza de Datos:** Los detalles correspondientes a este apartado se encuentran en el siguiente enlace [LIMPIEZA](../notebooks/002_LimpiezaEDA.ipynb)

- **4.2.Transformación de Características:** Los detalles correspondientes a este apartado se encuentran en el siguiente enlace [TRANSFORMACION](../notebooks/002_LimpiezaEDA.ipynb).

## 5.Selección y Entrenamiento del Modelo

- **Elección del Modelo:** Se toma como modelo GBC(Gradient Boosting Classifier), al ser el que mejor se comporta en la capacidaad de distinguir entre clases positivas y negativas, teniendo un ROC_AUC score de 0.635.
- **Configuración del Modelo:** El proceso de búsqueda de parámetros óptimos (GridSearchCV) puede haber contribuido a mejorar el rendimiento del GBC. Los parámetros óptimos encontrados durante la búsqueda, como {'learning_rate': 0.1, 'max_depth': 2, 'max_features': 3, 'n_estimators': 150}, sugieren una configuración que maximiza el área bajo la curva ROC.
- **Entrenamiento del Modelo:** El modelo Gradient Boosting Classifier (GBC) fue entrenado utilizando los siguientes parámetros óptimos:

learning_rate: 0.1
max_depth: 2
max_features: 3
n_estimators: 150

El conjunto de entrenamiento se utilizó para ajustar el modelo, donde se aplicó la técnica de boosting para mejorar iterativamente la capacidad predictiva del conjunto mediante la construcción de árboles débiles. El modelo resultante es una combinación ponderada de estos árboles, optimizada para maximizar el área bajo la curva ROC (ROC AUC Score).

## 6.Evaluación del Modelo

- **6.1Métricas de Evaluación y resultados:**

Durante el entrenamiento, se calcularon diversas métricas para evaluar el rendimiento del modelo:

Accuracy: 0.635 (63.5% de predicciones correctas).

Precision: 0.635 (63.5% de verdaderos positivos entre los positivos predichos).

Recall: 0.643 (64.3% de positivos reales identificados).

F1 Score: 0.639 (63.9% de equilibrio entre precision y recall).
ROC AUC Score: 0.635 (Área bajo la curva ROC).


Matriz de confusión

[[849 (verdaderos negativos), 503 (falsos positivos)],

 [486 (falsos negativos), 875 (verdaderos positivos)]]



## 7.Conclusiones y Recomendaciones

- **7.1Conclusiones:**

Mejor Modelo Seleccionado:

El modelo Gradient Boosting Classifier (GBC) con parámetros óptimos mostró el mejor rendimiento, con un ROC AUC Score de 0.635, indicando una sólida capacidad para discriminar entre clases positivas y negativas.

Desempeño Comparativo:

El GBC superó a otros modelos, como Regresión Logística, Árbol de Decisión, Random Forest y SVM, en términos de ROC AUC Score y otras métricas de evaluación.

Importancia de la Optimización de Hiperparámetros:

La búsqueda de hiperparámetros óptimos contribuyó significativamente al rendimiento del GBC. La configuración encontrada (learning_rate=0.1, max_depth=2, max_features=3, n_estimators=150) equilibró la complejidad del modelo y la prevención del sobreajuste.

Validación Cruzada:

La validación cruzada mostró consistencia en el rendimiento del GBC en diferentes particiones de datos, con un promedio de ROC AUC Score de 0.689 y una desviación estándar baja (0.012).


- **7.2Recomendaciones para Futuras Iteraciones:**

Explorar Nuevas Características:

Investigar la inclusión de nuevas características o la ingeniería de características podría mejorar aún más el rendimiento del modelo.

Evaluar Modelos Ensemble:

Considerar la implementación de modelos ensemble más complejos o la combinación de varios modelos para explorar sinergias y mejorar la robustez del sistema.

Monitoreo Continuo:

Establecer un proceso de monitoreo continuo del modelo en un entorno de producción para evaluar su rendimiento a medida que se reciben nuevos datos.

Interpretación de Resultados:

Profundizar en la interpretación de las características más relevantes identificadas por el GBC para obtener una comprensión más detallada de los factores que influyen en las predicciones.

Considerar Datos Adicionales:

Evaluar la posibilidad de incorporar datos adicionales o mejorar la calidad de los datos existentes para proporcionar al modelo una información más completa y precisa.

Optimización Continua de Hiperparámetros:

Realizar ajustes adicionales en la búsqueda de hiperparámetros para asegurar que el modelo esté completamente optimizado.

