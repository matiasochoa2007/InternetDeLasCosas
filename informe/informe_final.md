# Informe final · Clasificación automática de reseñas de Amazon (Hogar y Cocina)

**Materia:** Inteligencia de Datos — Universidad Iberoamericana León
**Métodos comparados:** KNN, Naive Bayes, SVM y Random Forest

> Todas las cifras de este informe provienen de los notebooks ejecutados y de los archivos en `resultados/` y `modelos/`.
> Las métricas de prueba se calcularon sobre **2,966 reseñas** (20 % del corpus limpio) contra las etiquetas generadas por reglas.

---

## 1. Definición del problema

Las estrellas son un criterio vago para entender qué quiere decir un cliente: dependen del punto de vista de cada persona y
las reseñas de 3 estrellas son muy mixtas. Leer todas las reseñas a mano no es viable cuando son miles. Por eso se planteó un
problema de **clasificación supervisada de texto** con tres etiquetas mutuamente excluyentes:

| Etiqueta | Descripción |
|---|---|
| **Producto** | La reseña menciona defectos, fallas o problemas relacionados con el producto. |
| **Envío** | La reseña menciona retrasos, daños durante el transporte o problemas de entrega. |
| **Sentimiento** | Expresa satisfacción o insatisfacción general sin centrarse en un problema específico. |

El objetivo es determinar cuál de los cuatro algoritmos ofrece el mejor desempeño, equilibrando calidad y costo computacional.

## 2. Recolección y selección de datos

- **Corpus:** 15,000 reseñas de Amazon de la categoría *Home_and_Kitchen*, en **inglés**, un archivo JSON por reseña,
  repartidas en `negative/` (1-2 estrellas), `neutral/` (3) y `positive/` (4-5), 5,000 por carpeta.
- **Variables conservadas:** `id`, `text` (ya incluye el título), `rating`, `polarity`, `category`, `asin`, `verified_purchase`.
  `rating` y `polarity` se usaron **solo para análisis**, nunca como características.
- **Limpieza** (`resultados/exploracion/bitacora_limpieza.csv`):

| Paso | Reseñas restantes | Eliminadas |
|---|---|---|
| Corpus inicial | 15,000 | — |
| Registros vacíos | 15,000 | 0 |
| Duplicados (texto normalizado) | 14,949 | 51 |
| Reseñas de ≤ 3 palabras | **14,826** | 123 |

- **Distribución inicial:** 4,947 negativas, 4,952 neutrales y 4,927 positivas (prácticamente balanceado por polaridad).
  Por estrellas: 1★ 2,997 · 2★ 1,950 · 3★ 4,952 · 4★ 993 · 5★ 3,934.
- **Longitud:** mediana de 43 palabras (media 70.5, máximo 1,797). Las reseñas neutrales son las más largas
  (mediana 53 palabras frente a 38 de las negativas y 39 de las positivas): mezclan elogios y quejas, lo que anticipa que serán las más difíciles.

## 3. Etiquetado de las reseñas

El protocolo plantea un etiquetado manual, pero etiquetar 14,826 reseñas a mano no es viable. Se usó un **etiquetado
semiautomático**:

1. **Reglas léxicas en inglés** (`reglas_etiquetado.py`, editables): 35 patrones fuertes de envío (*never arrived, arrived
   damaged, box was crushed, wrong item, lost in transit, poorly packaged…*), términos logísticos que solo cuentan con contexto
   negativo en la misma oración (*shipping/delivery/UPS/FedEx/USPS…* + *late, delayed, lost, damaged…*), 3 patrones de elogio al
   envío que se ignoran (*fast shipping, arrived on time…*) y 93 patrones de producto (*broke, defective, stopped working, flimsy,
   leaks, cracked, rust, smell, fell apart, doesn't fit, poor quality…*). Se manejan negaciones (*"no leaks"* no es defecto).
2. **Prioridad Envío > Producto > Sentimiento.** Envío es la clase más rara y la más accionable (logística), y cuando un artículo
   llega roto la reseña casi siempre usa también palabras de producto; si Producto tuviera prioridad, los daños de transporte
   nunca se detectarían. Producto va antes que Sentimiento porque un defecto concreto es más informativo que una opinión general.
3. **Validación manual:** se exportó `datos/muestra_validacion.csv` con **368 reseñas de la partición de prueba** (150 Producto,
   150 Sentimiento y las 68 de Envío disponibles en prueba, según las reglas), sin mostrar la etiqueta de las reglas ni las estrellas.
   El notebook 09 calcula el acuerdo (accuracy y kappa de Cohen) y evalúa los cuatro modelos contra esa etiqueta "oro".

**Distribución final de etiquetas** (`resultados/etiquetado/distribucion_etiquetas.csv`):

| Etiqueta | Reseñas | % |
|---|---|---|
| Envío | 342 | 2.31 % |
| Producto | 5,767 | 38.90 % |
| Sentimiento | 8,717 | 58.80 % |

**Etiqueta vs. polaridad** (confirma que las estrellas no bastan):

| Etiqueta | Negativa (1-2★) | Neutral (3★) | Positiva (4-5★) |
|---|---|---|---|
| Envío | 195 | 120 | 27 |
| Producto | 2,804 | 2,305 | 658 |
| Sentimiento | 1,948 | 2,527 | 4,242 |

Hay 658 reseñas positivas que mencionan un defecto concreto y 1,948 negativas que no describen ningún problema específico.
Las de 3 estrellas se reparten casi a la mitad entre Producto (2,305) y Sentimiento (2,527).

**Desbalance:** Envío es muy minoritaria (2.31 %). No se eliminó; se manejó con **F1 macro** como métrica principal,
`class_weight="balanced"` en SVM y Random Forest, `fit_prior=False` en Naive Bayes y un experimento de submuestreo (sección 9.4).

## 4. Preprocesamiento de texto

Función `utils.preprocesar_texto` (NLTK): normalización Unicode → minúsculas → eliminación de HTML (`<br />`, `&amp;`) y URLs →
expansión de contracciones negativas (*doesn't → does not*) → eliminación de puntuación, números y caracteres especiales →
reducción de letras repetidas (*soooo → soo*) → tokenización (`word_tokenize`) → etiquetado gramatical y **lematización WordNet**
→ eliminación de stopwords **conservando negaciones** (*not, no, never, nor, none, nothing, without…*) → limpieza de espacios.

Las negaciones se conservan porque cambian el significado (*"not work"* es un defecto; *"work"* no).

Ejemplo del protocolo (verificado con `assert` en el notebook 03):

> **"The package arrived late and damaged!!!" → "package arrive late damage"**

Tras el preprocesamiento ninguna reseña quedó vacía; la mediana bajó de 43 palabras a 22 tokens.

## 5. Vectorización TF-IDF

`TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.90, max_features=20000, sublinear_tf=True)`.
Con el conjunto de entrenamiento se obtienen **20,000 términos** (5,247 unigramas y 14,753 bigramas) y una matriz muy dispersa
(0.193 % de celdas distintas de cero). El vectorizador va **dentro de un `Pipeline`**, de modo que se ajusta solo con
entrenamiento (y, en la validación cruzada, solo con la parte de entrenamiento de cada pliegue): **no hay fuga de datos**.
Como referencia, ajustar solo el TF-IDF sobre las 11,860 reseñas de entrenamiento tomó 2.27 s (mediana de 3, notebook 08).

Términos más característicos por clase (TF-IDF promedio en la clase menos en el resto): Envío → *arrive, broken, arrive broken,
damage, come broken, packaging, receive*; Producto → *not, break, not work, smell, flimsy, cheap, thin*; Sentimiento → *love, great,
perfect, nice, beautiful, five star*.

## 6. División de datos

- **80 % entrenamiento (11,860) / 20 % prueba (2,966)**, estratificada por etiqueta, `random_state=42`, guardada en la columna
  `particion` para que los cuatro modelos usen exactamente la misma división.

| Etiqueta | Entrenamiento | Prueba |
|---|---|---|
| Envío | 274 (2.31 %) | 68 (2.29 %) |
| Producto | 4,613 (38.90 %) | 1,154 (38.91 %) |
| Sentimiento | 6,973 (58.79 %) | 1,744 (58.80 %) |

- **Validación cruzada estratificada de 5 pliegues** sobre entrenamiento para la búsqueda de hiperparámetros (`GridSearchCV`,
  `scoring="f1_macro"`). El conjunto de prueba solo se usó una vez, al final.

## 7. Entrenamiento de modelos

| Modelo | Configuración | Malla de búsqueda | Mejores hiperparámetros |
|---|---|---|---|
| **KNN** | distancia coseno, búsqueda exhaustiva | k ∈ {3, 5, 7, 11, 15, 21, 31}; weights ∈ {uniform, distance} | k = 5, uniform |
| **Naive Bayes** | MultinomialNB, `fit_prior=False` | alpha ∈ {0.01, 0.03, 0.1, 0.3, 0.5, 1.0} | alpha = 0.1 |
| **SVM** | `class_weight="balanced"` | LinearSVC C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3}; SVC RBF C ∈ {1, 10} | **LinearSVC, C = 0.3** |
| **Random Forest** | `class_weight="balanced"`, `n_jobs=1` | n_estimators ∈ {200, 400}; max_depth ∈ {None, 60}; min_samples_leaf ∈ {1, 2, 5} | 200 árboles, sin límite de profundidad, min_samples_leaf = 2 |

En SVM, el kernel RBF (F1 macro en CV de 0.6624 con C = 10 y 0.6546 con C = 1) quedó por debajo del lineal (0.7115) y fue
mucho más lento, por lo que se eligió `LinearSVC`. Cada modelo se guardó con `joblib` en `modelos/`.

## 8. Evaluación de resultados

### 8.1 Métricas en prueba por modelo

| Modelo | Accuracy | Precisión macro | Recall macro | **F1 macro** | F1 ponderado | Errores |
|---|---|---|---|---|---|---|
| KNN | 0.7063 | 0.6598 | 0.5264 | 0.5586 | 0.6912 | 871 |
| Naive Bayes | 0.7202 | 0.5940 | 0.6123 | 0.5955 | 0.7237 | 830 |
| **SVM** | **0.8240** | **0.7681** | **0.6955** | **0.7241** | **0.8225** | **522** |
| Random Forest | 0.7839 | 0.6762 | 0.6860 | 0.6807 | 0.7852 | 641 |

**F1 por clase:**

| Modelo | Envío | Producto | Sentimiento |
|---|---|---|---|
| KNN | 0.3191 | 0.5723 | 0.7845 |
| Naive Bayes | 0.3310 | 0.7024 | 0.7532 |
| SVM | **0.5310** | **0.7782** | **0.8631** |
| Random Forest | 0.4748 | 0.7399 | 0.8272 |

Envío es la clase más difícil para todos (pocos ejemplos). En el SVM, Envío tiene precisión 0.6667 y recall 0.4412: cuando
predice Envío suele acertar, pero se le escapa más de la mitad de los casos.

**Validación cruzada (F1 macro, 5 pliegues):** KNN 0.5720 ± 0.0270 · Naive Bayes 0.6248 ± 0.0094 · SVM 0.7115 ± 0.0126 ·
Random Forest 0.6973 ± 0.0234.

**Tiempos** (mediana de 3 repeticiones, un núcleo para el clasificador; incluyen el ajuste del TF-IDF):

| Modelo | Entrenamiento (s) | Predicción de 2,966 reseñas (s) | Búsqueda de hiperparámetros (s) |
|---|---|---|---|
| KNN | 1.17 | 0.71 | 44.8 |
| Naive Bayes | 2.11 | 0.20 | 17.7 |
| SVM | 2.30 | 0.20 | 192.9 |
| Random Forest | 15.79 | 0.80 | 227.8 |

- **Random Forest** es claramente el más costoso de entrenar (15.79 s, unas 7 veces más que el SVM) y el más
  lento al predecir (0.80 s).
- **KNN** casi no "entrena" (solo guarda los vectores), pero al predecir compara cada reseña nueva contra las 11,860 de
  entrenamiento: tarda 0.71 s, más de 3 veces lo que tardan Naive Bayes y SVM.
- **Naive Bayes y SVM** son los más baratos (alrededor de 2 s para entrenar y 0.2 s para predecir).
- La búsqueda de hiperparámetros del SVM (192.9 s) es larga solo por la prueba opcional con kernel RBF, cuyos ajustes
  tardaron en promedio entre 97 y 155 s frente a unos 3-4 s del lineal.
- Los tiempos de pocos segundos varían entre ejecuciones según la carga del equipo (el TF-IDF solo, medido aparte, tomó
  2.27 s, más que el pipeline completo de KNN medido antes); diferencias de alrededor de un segundo no son significativas.
  En una primera corrida, sin cerrar los procesos auxiliares de `GridSearchCV` antes de medir, todos los modelos marcaban ~7 s de
  entrenamiento; por eso se cambió la medición (mediana de 3 repeticiones) y se volvieron a ejecutar los notebooks.

Las matrices de confusión (gráfica y CSV) están en `resultados/<modelo>/` y lado a lado en
`resultados/comparacion/matrices_confusion.png`.

### 8.2 Análisis de errores

**Tipos de error** (suma de los cuatro modelos, `resultados/comparacion/errores_por_tipo.csv`):

| Real → Predicha | KNN | Naive Bayes | SVM | Random Forest | Total |
|---|---|---|---|---|---|
| Sentimiento → Producto | 228 | 529 | 214 | 318 | 1,289 |
| Producto → Sentimiento | 579 | 204 | 255 | 250 | 1,288 |
| Envío → Producto | 30 | 37 | 31 | 30 | 128 |
| Sentimiento → Envío | 2 | 34 | 7 | 16 | 59 |
| Producto → Envío | 9 | 19 | 8 | 22 | 58 |
| Envío → Sentimiento | 23 | 7 | 7 | 5 | 42 |

- La confusión dominante es **Producto ↔ Sentimiento**: en el SVM son 469 de sus 522 errores (89.8 %). Son reseñas que mezclan una
  opinión general con un defecto leve (*"It's OK. Good deal for the price, but… it's cheaply constructed"*, etiquetada Producto y
  predicha Sentimiento) o que describen problemas con palabras que no están en el léxico.
- **KNN** tiende a predecir Sentimiento (la clase mayoritaria): 579 reseñas de Producto acabaron como Sentimiento.
  **Naive Bayes** hace lo contrario (529 Sentimiento → Producto) por usar priors uniformes.
- **Envío → Producto** es el error más común en Envío: reseñas donde el daño de transporte se describe junto a quejas de calidad
  (*"Cheap, broken, and overpriced!… It arrived bent"*).
- Varios "errores" son en realidad **errores de las reglas** que el modelo corrige: *"Missing a leg. Arrived with only 3 legs"*
  quedó como Sentimiento por las reglas, pero el SVM la predijo como Envío, lo cual es razonable. Esto refuerza la necesidad de la
  validación manual.

**Palabras más influyentes** (coeficientes del SVM, `resultados/svm/palabras_influyentes.csv`):
Envío → *arrive, packaging, deliver, damage, come broken, receive, not receive, dent, miss*;
Producto → *not work, break, smell, flimsy, not fit, useless, leak, not hold, not stay*;
Sentimiento → *no, perfect, great, love, perfectly, five star, nice*.
En Naive Bayes destacan bigramas como *arrive broken, come damage, arrive late, never receive* (Envío) y *not stay, not work,
not fit, stink* (Producto). En Random Forest los términos más importantes son *arrive, broken, come, break, not, packaging*.
Que *not work*, *not fit* o *no* estén entre los más influyentes confirma que **conservar las negaciones fue útil**; que
aparezcan las mismas palabras del léxico confirma que los modelos **aprenden en buena parte las reglas** (ver limitaciones).

### 8.3 ¿Se clasifican peor las reseñas de 3 estrellas?

**Sí.** El F1 macro más bajo de los cuatro modelos corresponde a las reseñas neutrales:

| Modelo | F1 macro negativa | F1 macro neutral (3★) | F1 macro positiva | Accuracy neutral |
|---|---|---|---|---|
| KNN | 0.5386 | **0.5200** | 0.5203 | 0.6380 |
| Naive Bayes | 0.5621 | **0.5067** | 0.5082 | 0.6507 |
| SVM | 0.7253 | **0.6646** | 0.7432 | 0.7661 |
| Random Forest | 0.6935 | **0.5942** | 0.6359 | 0.7123 |

Para el SVM la accuracy pasa de 0.9039 en positivas y 0.8031 en negativas a 0.7661 en neutrales. Esto coincide con la
motivación del proyecto: las reseñas de 3 estrellas mezclan elogios y quejas, y son justamente las que más se benefician de
una clasificación por tipo de problema en lugar de por estrellas.

### 8.4 Validación con etiqueta manual

**Pendiente.** La muestra `datos/muestra_validacion.csv` (368 reseñas de prueba) se entrega lista para etiquetar a mano. Al
llenarla y ejecutar `09_validacion_manual.ipynb` se obtendrán: el acuerdo reglas-vs-manual (accuracy y kappa de Cohen) y el
desempeño de los cuatro modelos contra la etiqueta humana (`resultados/validacion_manual/`). Estas cifras deben agregarse aquí;
son las que dicen qué tan confiables son realmente las métricas de la sección 8.1.

## 9. Comparación de algoritmos

### 9.1 Tabla comparativa (`resultados/comparacion/tabla_comparativa.csv` y `.md`)

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (ponderado) | F1 Envío | F1 Producto | F1 Sentimiento | F1 macro CV (media) | F1 macro CV (desv. est.) | Brecha CV - prueba | Errores en prueba | Tiempo búsqueda (s) | Tiempo entrenamiento (s) | Tiempo predicción (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KNN | 0.7063 | 0.6598 | 0.5264 | 0.5586 | 0.6912 | 0.3191 | 0.5723 | 0.7845 | 0.5720 | 0.0270 | 0.0134 | 871 | 44.8065 | 1.1738 | 0.7097 |
| Naive Bayes | 0.7202 | 0.5940 | 0.6123 | 0.5955 | 0.7237 | 0.3310 | 0.7024 | 0.7532 | 0.6248 | 0.0094 | 0.0292 | 830 | 17.6693 | 2.1131 | 0.1955 |
| SVM | 0.8240 | 0.7681 | 0.6955 | 0.7241 | 0.8225 | 0.5310 | 0.7782 | 0.8631 | 0.7115 | 0.0126 | -0.0126 | 522 | 192.9144 | 2.3042 | 0.1992 |
| Random Forest | 0.7839 | 0.6762 | 0.6860 | 0.6807 | 0.7852 | 0.4748 | 0.7399 | 0.8272 | 0.6973 | 0.0234 | 0.0166 | 641 | 227.8205 | 15.7856 | 0.7972 |

*Brecha = F1 macro CV − F1 macro prueba. Tiempos en segundos (time.perf_counter, mediana de 3).*

### 9.2 Ranking por criterio (1 = mejor)

| Modelo | F1 macro | Generalización |brecha| | Errores | Estabilidad (desv. CV) | Tiempo entrenamiento | Suma de posiciones |
|---|---|---|---|---|---|---|
| KNN | 4 | 2 | 4 | 4 | 1 | 15 |
| Naive Bayes | 3 | 4 | 3 | 1 | 2 | 13 |
| SVM | 1 | 1 | 1 | 2 | 3 | 8 |
| Random Forest | 2 | 3 | 2 | 3 | 4 | 14 |

SVM tiene la menor suma de posiciones (8) y es primero en F1 macro, generalización y errores.

### 9.3 Gráficas

`resultados/comparacion/`: `metricas_comparadas.png` (accuracy, precisión, recall y F1 macro; F1 por clase),
`cv_vs_prueba.png` (generalización), `tiempos.png` (costo computacional, escala log), `matrices_confusion.png`
y `accuracy_por_polaridad.png`.

### 9.4 Experimento de submuestreo

Se reentrenó cada modelo con sus mismos hiperparámetros sobre un entrenamiento balanceado (274 reseñas por clase, 822 en total)
y se evaluó sobre la misma prueba:

| Modelo | F1 macro original | F1 macro submuestreo | Recall Envío original | Recall Envío submuestreo |
|---|---|---|---|---|
| KNN | 0.5586 | 0.4133 | 0.2206 | 0.6912 |
| Naive Bayes | 0.5955 | 0.4864 | 0.3529 | 0.7647 |
| SVM | 0.7241 | 0.5457 | 0.4412 | 0.8235 |
| Random Forest | 0.6807 | 0.5548 | 0.4853 | 0.8824 |

El submuestreo **sube mucho el recall de Envío** (en el SVM de 0.4412 a 0.8235) pero **baja el F1 macro en todos los modelos**
(el SVM pierde 0.1784) porque descarta más del 90 % de los datos de Producto y Sentimiento y genera muchas falsas alarmas de Envío.
Por eso se conservan los modelos entrenados con todos los datos y pesos por clase. Si en la práctica fuera prioritario no dejar
pasar ningún problema de envío (aunque haya más falsas alarmas), el submuestreo sería una alternativa a considerar.

## 10. Selección del modelo óptimo

**Modelo elegido: SVM lineal (`LinearSVC`, C = 0.3, `class_weight="balanced"`)**, guardado en `modelos/mejor_modelo.joblib`.

Regla fijada antes de ver los resultados: mayor F1 macro en prueba; empate práctico (< 0.005) → menor tiempo de entrenamiento.
Ningún otro modelo quedó dentro de esa tolerancia. Cumple los cuatro criterios del protocolo:

- **Mayor F1:** F1 macro 0.7241, 0.0434 por encima de Random Forest (0.6807), 0.1286 sobre Naive Bayes y 0.1655 sobre KNN.
  También tiene el mejor F1 en cada una de las tres clases, incluida Envío (0.5310).
- **Mejor generalización:** F1 macro en CV 0.7115 frente a 0.7241 en prueba (brecha −0.0126, la menor en valor absoluto):
  el resultado de prueba es consistente con la validación cruzada, sin señales de sobreajuste.
- **Menos errores:** 522 de 2,966 (17.6 %), frente a 641 de Random Forest, 830 de Naive Bayes y 871 de KNN.
- **Estabilidad:** desviación estándar entre pliegues de 0.0126, segunda menor (Naive Bayes tiene 0.0094 pero con un F1 mucho más bajo).
- **Costo computacional:** entrena en 2.30 s y clasifica las 2,966 reseñas de prueba en 0.20 s, del mismo orden
  que Naive Bayes y KNN (las diferencias de alrededor de un segundo están dentro de la variación entre mediciones) y muy por debajo de
  Random Forest (15.79 s de entrenamiento). En el ranking queda tercero en tiempo de entrenamiento, pero es el
  mejor balance entre rendimiento y costo: los modelos más baratos tienen entre 0.13 y 0.17 menos de F1 macro.

La función `utils.clasificar_resena(texto)` carga este modelo y devuelve la etiqueta y su confianza (softmax de la función de
decisión del SVM, que ordena qué tan segura es la predicción pero no es una probabilidad calibrada). Ejemplos
(`resultados/comparacion/ejemplos_clasificar_resena.csv`):

| Reseña | Etiqueta | Confianza | Envío | Producto | Sentimiento |
|---|---|---|---|---|---|
| The package arrived two weeks late and the box was completely crushed. | **Envío** | 0.8179 | 0.8179 | 0.0744 | 0.1077 |
| The handle snapped off after three uses and the pan started to rust. | **Producto** | 0.7066 | 0.1018 | 0.7066 | 0.1916 |
| Absolutely love this blanket, so cozy and the color is beautiful! | **Sentimiento** | 0.8288 | 0.0648 | 0.1064 | 0.8288 |
| It's okay. Does the job, nothing special for the price. | **Sentimiento** | 0.8075 | 0.0701 | 0.1224 | 0.8075 |
| UPS left it at the wrong address and it arrived with a broken lid. | **Envío** | 0.8986 | 0.8986 | 0.0248 | 0.0766 |

## 11. Conclusiones

**Qué algoritmo obtuvo mejores resultados.** El **SVM lineal** fue el mejor en todas las métricas de calidad (accuracy 0.8240,
F1 macro 0.7241, F1 ponderado 0.8225) con un costo computacional bajo, del mismo orden que Naive Bayes y KNN. Random Forest quedó segundo (F1 macro 0.6807) con un
costo de entrenamiento mayor. Naive Bayes es muy rápido pero confunde mucho Sentimiento con Producto, y KNN fue el más débil
(F1 macro 0.5586), lo esperable en espacios TF-IDF de 20,000 dimensiones donde la noción de "vecino cercano" pierde fuerza.

**Qué errores fueron más frecuentes.** La confusión **Producto ↔ Sentimiento** concentra la gran mayoría de los errores (cerca
del 90 % en el SVM), sobre todo en reseñas de **3 estrellas**, que mezclan opinión general y defectos menores. En Envío el error
típico es confundirlo con Producto cuando el daño de transporte se describe con palabras de calidad.

**Limitaciones del dataset y del método.**
- **Corpus en inglés y de una sola categoría** (*Home_and_Kitchen*): los léxicos, el preprocesamiento y los modelos no se
  pueden aplicar directamente a reseñas en español ni a otras categorías sin adaptarlos y reentrenarlos.
- **Etiquetas generadas con reglas.** Los modelos se entrenaron y evaluaron contra etiquetas automáticas. Como las reglas se basan
  en palabras clave y los modelos aprenden pesos sobre esas mismas palabras (ver palabras influyentes), las métricas pueden estar
  **infladas**: miden qué tan bien el modelo imita a las reglas, no necesariamente qué tan bien entiende la reseña. Además, las
  reglas tienen errores propios (por ejemplo, reseñas de Envío no detectadas que quedaron como Sentimiento). Por eso la
  **validación manual** (sección 8.4) es indispensable antes de usar el clasificador en la práctica.
- **Desbalance de clases:** Envío es solo el 2.31 % del corpus (68 reseñas en prueba), lo que limita su F1 (0.5310 en el mejor
  modelo) y hace que sus métricas tengan alta variabilidad.
- **Una sola etiqueta por reseña:** muchas reseñas reales mencionan producto **y** envío; la prioridad Envío > Producto >
  Sentimiento resuelve el conflicto pero pierde información.
- **TF-IDF es una bolsa de palabras:** no entiende orden ni contexto más allá de bigramas (ironía, comparaciones, negaciones lejanas).

**Mejoras futuras.**
- Terminar y ampliar el **etiquetado manual** (o con varias personas para medir acuerdo entre anotadores) y reentrenar con esas etiquetas.
- **Word embeddings** (Word2Vec, GloVe, fastText) para capturar sinónimos que no están en el léxico.
- Modelos de **Deep Learning** y **transformers tipo BERT** (por ejemplo DistilBERT o RoBERTa afinados), que entienden contexto y negación.
- **Clasificación multietiqueta** (una reseña puede ser Producto y Envío a la vez).
- Más datos de Envío (otras categorías o sobremuestreo/aumentación de texto) y calibración de probabilidades del SVM
  (`CalibratedClassifierCV`) para dar una confianza interpretable.
