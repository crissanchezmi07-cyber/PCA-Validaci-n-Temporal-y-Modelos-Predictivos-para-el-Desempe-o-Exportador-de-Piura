# PCA Validación Temporal y Modelos Predictivos para el Desempeño Exportador de Piura

El objetivo de este proyecto es analizar si el crédito (variable económica clave) está relacionado con el comportamiento de las exportaciones en la región de Piura. Para ello combinamos técnicas tradicionales (regresiones con validación temporal) con herramientas modernas de Machine Learning, que permiten ampliar el análisis y desarrollar interpretaciones complementarias más robustas.

El enfoque del trabajo sigue tres ejes:

  1. Comprender la estructura económica del dataset

  2. Construir modelos de regresión orientados a series de tiempo

  3. Incorporar un módulo adicional de clasificación como complemento, no como parte central del análisis económico, buscando enriquecer la lectura del fenómeno exportador.

**Preparación de datos e ingeniería de características**

Realizamos un trabajo cuidadoso de curación y transformación de datos, ya que las series económicas suelen presentar tendencias, necesitan rezagos, pueden tener multicolinealidad, y deben respetar el orden temporal para evitar filtraciones de información.

Variables originales utilizadas:

a. Exportaciones de Piura

b. Exportaciones nacionales

c. Crédito total, en soles y dólares

d. Crédito nacional

e. Tipo de cambio

f. Riesgo país (EMBIG)

g. ICEN e ICEN_exp (Fenómeno El Niño)

  Transformaciones justificadas

Ratio exportaciones Piura/nacional → mide el desempeño relativo de Piura frente al país.

Variaciones mensuales → ayudan a capturar comportamiento dinámico.

Rezagos → permiten que los modelos aprendan dependencias temporales.

Eliminación de NA y estandarización según se requiera.

Estas transformaciones son esenciales para evitar conclusiones engañosas debido a tendencias o escalas distintas entre variables.

**PCA: Diagnóstico de estructura y multicolinealidad**

Permite identificar grupos de variables altamente correlacionadas.

Ayuda a entender qué factores explican mayor variabilidad en el sistema económico.

Facilita la decisión sobre mantener ciertas variables, crear rezagos o transformarlas.

Proporciona un biplot que da una lectura visual intuitiva.

En un contexto económico, esto es valioso porque muchas variables se mueven juntas (por ejemplo, créditos y exportaciones), y PCA ayuda a visualizar estas relaciones.

**Modelos de regresión para series de tiempo**

Se parte de df_model, que es el DataFrame ya limpio (sin NaN) y ordenado por fecha.

Se define la variable objetivo:
target = "exportaciones_piura"
y = df_model[target]

Se definen las variables explicativas:
X = df_model.drop(columns=[target])

Se configura el esquema de validación temporal:
tscv = TimeSeriesSplit(n_splits=5)
para partir los datos en pliegues que respetan el orden del tiempo.

Ridge y Lasso (modelos lineales regularizados)

Importa RidgeCV y LassoCV, junto con StandardScaler, Pipeline y mean_squared_error.
Define un conjunto de valores de α (alphas = np.logspace(-3, 3, 50)), que son los posibles niveles de regularización.

Construye pipelines para:
Escalar las variables (StandardScaler),
Ajustar un modelo RidgeCV y un modelo LassoCV,
Usando el esquema TimeSeriesSplit como validación.
Para cada modelo, calcula el MSE promedio sobre los pliegues de TimeSeriesSplit.
Compara los MSE de Ridge y Lasso, se queda con el que tiene menor MSE y lo guarda como mejor_modelo y sus predicciones como mejor_pred.
Finalmente, genera un gráfico de “Predicción vs Real” para ese mejor modelo lineal:

Línea real: exportaciones de Piura.
Línea predicha: estimación del modelo.

Se prueba primero si con una combinación lineal de las variables (pero regularizada para evitar inestabilidad) se puede explicar bien el nivel de exportaciones.
El gráfico permite ver visualmente qué tan bien el modelo sigue la trayectoria real.

Random Forest y XGBoost (modelos no lineales)

Importa los modelos no lineales (RandomForestRegressor, XGBRegressor, etc.) y GridSearchCV.
Define rejillas de hiperparámetros (número de árboles, profundidad máxima, etc.) para cada modelo.
Usa GridSearchCV junto con TimeSeriesSplit:
Para cada combinación de hiperparámetros, entrena el modelo, calcula el MSE en validación temporal, y se queda con la mejor combinación según el MSE.
Compara el mejor Random Forest y el mejor XGBoost, se queda con el que tiene menor MSE (en tu caso, XGBoost).
Calcula la importancia de variables de ese mejor model (best_model.feature_importances_) y construye una tabla ordenando las variables de mayor a menor importancia.

Se prueba si un modelo más flexible, que permite no linealidades e interacciones, mejora la capacidad de predicción frente a los modelos lineales.
La tabla de importancias dice qué variables pesan más a la hora de explicar las exportaciones de Piura.

Tabla comparativa de MSE

En el bloque donde se construye:
XGBoost obtiene un MSE mucho menor que Ridge. Por lo tanto, es el modelo elegido.

Con números concretos se muestra que el modelo no lineal XGBoost ajusta mejor la serie de exportaciones que el modelo lineal Ridge. Eso sugiere que el fenómeno tiene componentes no lineales o interacciones que XGBoost sí capta.


**Análisis complementario: Modelos de clasificación**

Aunque el objetivo principal es regresión, añadimos un módulo de clasificación porque puede ser útil para identificar probabilísticamente periodos donde el desempeño relativo de Piura mejora, generar señales/alertas basadas en probabilidades, complementar la regresión con una lectura diferente del comportamiento del ratio.

Definición de la variable binaria
Se construyó:
y = 1 si el ratio Piura/Nacional mejora
y = 0 si no mejora

Modelos evaluados
Logit: El Logit produce probabilidades razonables, pero suavizadas; no capta bien todos los saltos, lo cual se esperaba debido a la inestabilidad del ratio.
Random Forest Classifier: RF produce probabilidades más sensibles a los cambios y captura mejor la dirección del movimiento del ratio.
No se pretende reemplazar el análisis del OLS, sino mostrar cómo un modelo de clasificación interpreta la dinámica del ratio

Desbalance y técnicas aplicadas
Como los datos no tienen el mismo número de mejoras y caídas, se utilizó SMOTE, que:
Crea observaciones sintéticas,
Balancea las clases,
Mejora la estabilidad del entrenamiento.

Ajuste umbral basado en costos 
Económicamente, equivocar un FN es más costoso que equivocarse en un FP. 
Por eso: 
FN:0.8
FP:0.2
Se buscó el umbral que minimiza el costo total, y se graficó sobre la curva Precision-Recall. 

Este apartado no pretende predecir perfectos saltos mes a mes, sino: mostrar si hay señales probabilísticas coherentes, analizar la tendencia de mejora del ratio, complementar el análisis principal con herramientas modernas de clasificación. 

**Conclusiones**

El modelo XGBoost fue el mejor predictor de las exportaciones de Piura, obteniendo el menor MSE en validación temporal. Esto significa que XGBoost logró ajustarse de forma más precisa a la dinámica mensual de las exportaciones que los modelos lineales regularizados (Ridge o Lasso). Su mejor rendimiento indica la presencia de relaciones no lineales y posibles interacciones entre crédito, tipo de cambio, riesgo país e indicadores climáticos.

El modelo Ridge presentó un desempeño aceptable, capaz de capturar la tendencia general de la serie, pero con errores mayores respecto a XGBoost. Esto confirma que existe una parte importante del comportamiento exportador que no puede explicarse por una combinación lineal simple de las variables económicas.

En los gráficos de predicción vs valor real, XGBoost logra reproducir mejor los movimientos ascendentes y descendentes de las exportaciones, especialmente en periodos de cambios moderados. Aunque ningún modelo es capaz de predecir con exactitud los picos o caídas más abruptas, el comportamiento general sí es capturado de forma adecuada.

El análisis de importancia de variables muestra que factores como el crédito total, el tipo de cambio, el riesgo país y los indicadores climáticos (ICEN, ICEN_exp) tienen un peso significativo en la explicación del nivel exportador. Esto aporta evidencia clara de que el desempeño de las exportaciones no depende únicamente de los flujos de crédito, sino que es una variable multifactorial.

La validación mediante TimeSeriesSplit demostró que el modelo mantiene estabilidad fuera de la muestra y no depende de información futura para predecir el pasado. Esto refuerza la credibilidad de los resultados y confirma que los errores reportados son representativos.
