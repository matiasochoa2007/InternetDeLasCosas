# Clasificación automática de reseñas de Amazon (Hogar y Cocina)

Proyecto de **Inteligencia de Datos** — Universidad Iberoamericana León.

Se construye un clasificador que asigna cada reseña de Amazon (categoría *Home_and_Kitchen*, en inglés) a **una** de tres etiquetas
y se comparan cuatro métodos —**KNN, Naive Bayes, SVM y Random Forest**— para identificar el más óptimo, siguiendo el protocolo
`InicioProyectoID.pdf` (secciones 1-11).

| Etiqueta | Significado |
|---|---|
| **Producto** | La reseña menciona defectos, fallas o problemas del producto. |
| **Envío** | La reseña menciona retrasos, daños durante el transporte o problemas de entrega. |
| **Sentimiento** | Satisfacción o insatisfacción general, sin centrarse en un problema específico de producto o envío. |

El informe completo está en [`informe/informe_final.md`](informe/informe_final.md).

---

## 1. Estructura de la carpeta

```
proyecto_amazon/
├── README.md                         <- este archivo
├── requirements.txt                  <- dependencias
├── utils.py                          <- funciones compartidas (carga, preprocesamiento, TF-IDF, métricas, clasificar_resena)
├── reglas_etiquetado.py              <- léxicos y reglas editables para el etiquetado semiautomático
├── 01_carga_y_limpieza.ipynb         <- lectura de los 15,000 JSON, limpieza, distribución y longitud
├── 02_etiquetado_y_particion.ipynb   <- etiquetado por reglas, partición 80/20, muestra de validación
├── 03_preprocesamiento_y_tfidf.ipynb <- limpieza de texto, lematización, TF-IDF
├── 04_knn.ipynb                      <- KNN (coseno)
├── 05_naive_bayes.ipynb              <- MultinomialNB
├── 06_svm.ipynb                      <- LinearSVC (+ SVC RBF opcional)
├── 07_random_forest.ipynb            <- Random Forest
├── 08_comparacion_y_seleccion.ipynb  <- tabla comparativa, gráficas, selección, submuestreo, clasificar_resena
├── 09_validacion_manual.ipynb        <- acuerdo reglas vs. etiqueta manual y modelos vs. etiqueta manual
├── datos/
│   ├── corpus_amazon/                <- copia descomprimida del corpus (el .zip original no se toca)
│   ├── resenas_limpias.csv           <- salida de 01
│   ├── resenas_etiquetadas.csv       <- salida de 02 (etiqueta_reglas, regla_disparada, particion)
│   ├── resenas_procesadas.csv        <- salida de 03 (texto_procesado) — la usan 04-09
│   └── muestra_validacion.csv        <- 368 reseñas de PRUEBA para etiquetar a mano
├── resultados/
│   ├── exploracion/ etiquetado/ preprocesamiento/
│   ├── knn/ naive_bayes/ svm/ random_forest/   <- métricas, reporte, matriz de confusión (PNG y CSV),
│   │                                              predicciones_prueba.csv, errores, palabras influyentes
│   ├── comparacion/                  <- tabla_comparativa.csv / .md, gráficas, ranking, submuestreo
│   └── validacion_manual/            <- se llena al correr 09 con la muestra etiquetada
├── modelos/                          <- knn.joblib, naive_bayes.joblib, svm.joblib, random_forest.joblib,
│                                        mejor_modelo.joblib, mejor_modelo_info.json
└── informe/informe_final.md
```

## 2. Instalación y ejecución

Requisitos: Python 3.10+ (se probó con Python 3.12.10 en Windows 11).

```powershell
python -m pip install -r requirements.txt
```

Los recursos de NLTK (stopwords, WordNet, punkt y el etiquetador gramatical) se descargan solos la primera vez
(`utils.preparar_nltk()`).

Los notebooks se ejecutan **en orden** (01 → 09) desde esta carpeta, en Jupyter o con:

```powershell
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 01_carga_y_limpieza.ipynb
```

Tiempos aproximados en la máquina de prueba (16 núcleos): 01 y 02 ≈ 2-4 min cada uno, 03 ≈ 5 min (etiquetado gramatical),
04 ≈ 4 min, 05 ≈ 1 min, 06 ≈ 4 min (incluye la prueba opcional con kernel RBF), 07 ≈ 6 min, 08 ≈ 1 min, 09 < 1 min.
Pueden ser bastante mayores si el equipo está ocupado o en ahorro de energía.
Todos los notebooks se entregan ya ejecutados y con sus salidas.

> En Windows, la consola puede mostrar un `KeyError ... joblib_memmapping_folder` al terminar un `GridSearchCV`.
> Es un aviso inofensivo de limpieza de archivos temporales de joblib; no afecta los resultados.

## 3. Resultados principales

**Distribución final de etiquetas** (reglas, 14,826 reseñas): Envío 342 (2.31 %), Producto 5,767 (38.90 %), Sentimiento 8,717 (58.80 %). Partición 80/20 estratificada: 11,860 entrenamiento / 2,966 prueba.

**Tabla comparativa en prueba** (`resultados/comparacion/tabla_comparativa.csv`; métricas contra etiquetas de reglas):

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (ponderado) | F1 Envío | F1 Producto | F1 Sentimiento | F1 macro CV (media) | F1 macro CV (desv. est.) | Brecha CV - prueba | Errores en prueba | Tiempo búsqueda (s) | Tiempo entrenamiento (s) | Tiempo predicción (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KNN | 0.7063 | 0.6598 | 0.5264 | 0.5586 | 0.6912 | 0.3191 | 0.5723 | 0.7845 | 0.5720 | 0.0270 | 0.0134 | 871 | 44.8065 | 1.1738 | 0.7097 |
| Naive Bayes | 0.7202 | 0.5940 | 0.6123 | 0.5955 | 0.7237 | 0.3310 | 0.7024 | 0.7532 | 0.6248 | 0.0094 | 0.0292 | 830 | 17.6693 | 2.1131 | 0.1955 |
| SVM | 0.8240 | 0.7681 | 0.6955 | 0.7241 | 0.8225 | 0.5310 | 0.7782 | 0.8631 | 0.7115 | 0.0126 | -0.0126 | 522 | 192.9144 | 2.3042 | 0.1992 |
| Random Forest | 0.7839 | 0.6762 | 0.6860 | 0.6807 | 0.7852 | 0.4748 | 0.7399 | 0.8272 | 0.6973 | 0.0234 | 0.0166 | 641 | 227.8205 | 15.7856 | 0.7972 |

**Modelo recomendado: SVM lineal** (`LinearSVC`, C = 0.3, `class_weight="balanced"`), guardado en `modelos/mejor_modelo.joblib`: mayor F1 macro (0.7241), mejor F1 en las tres clases, menor brecha entre CV y prueba (−0.0126), menos errores (522 de 2,966) y bajo costo. La confusión más común en todos los modelos es Producto ↔ Sentimiento, y las reseñas de 3 estrellas son las que peor se clasifican. Detalle y justificación en el [informe](informe/informe_final.md).

## 4. Función `clasificar_resena(texto)`

```python
from utils import clasificar_resena
clasificar_resena("The package arrived two weeks late and the box was completely crushed.")
# {'etiqueta': 'Envío', 'confianza': ..., 'puntajes': {'Envío': ..., 'Producto': ..., 'Sentimiento': ...},
#  'tipo_confianza': ...}
```

Carga `modelos/mejor_modelo.joblib`, preprocesa el texto exactamente igual que en el entrenamiento y devuelve la etiqueta,
la confianza de esa etiqueta y los puntajes de las tres clases.

Los 5 ejemplos de prueba (notebook 08):

| Reseña | Etiqueta | Confianza | Envío | Producto | Sentimiento |
|---|---|---|---|---|---|
| The package arrived two weeks late and the box was completely crushed. | **Envío** | 0.8179 | 0.8179 | 0.0744 | 0.1077 |
| The handle snapped off after three uses and the pan started to rust. | **Producto** | 0.7066 | 0.1018 | 0.7066 | 0.1916 |
| Absolutely love this blanket, so cozy and the color is beautiful! | **Sentimiento** | 0.8288 | 0.0648 | 0.1064 | 0.8288 |
| It's okay. Does the job, nothing special for the price. | **Sentimiento** | 0.8075 | 0.0701 | 0.1224 | 0.8075 |
| UPS left it at the wrong address and it arrived with a broken lid. | **Envío** | 0.8986 | 0.8986 | 0.0248 | 0.0766 |

## 5. Lo que te toca hacer: validación manual

1. Abre `datos/muestra_validacion.csv` (Excel o Google Sheets). Tiene **368 reseñas de la partición de prueba**
   (150 que las reglas marcaron como Producto, 150 como Sentimiento y las **68** de Envío que hay en prueba), revueltas.
   La etiqueta de las reglas y las estrellas están **ocultas** a propósito para no sesgarte.
2. Lee cada reseña y escribe en `etiqueta_manual`: `Producto`, `Envío` o `Sentimiento` (también sirven `P`, `E`, `S`,
   `envio`...). La columna `comentario` es opcional. Puedes dejar filas vacías: solo se usan las llenas.
3. Guarda como **CSV UTF-8** con el mismo nombre.
4. Ejecuta `09_validacion_manual.ipynb` completo. Obtendrás:
   - acuerdo reglas vs. manual (**accuracy y kappa de Cohen**) y matriz de confusión;
   - desempeño de los **4 modelos contra tu etiqueta** (accuracy, F1 macro, kappa, F1 de Envío) comparado con su desempeño
     contra las reglas en la misma muestra (`resultados/validacion_manual/`).
5. Agrega esas cifras a la sección 8.4 del informe (está marcada como pendiente).

Si vuelves a correr el notebook 02 después de llenar la muestra, **no** se sobrescribe tu archivo: la muestra nueva se guarda
como `muestra_validacion_nueva.csv`.

## 6. Decisiones y supuestos

**Datos y limpieza**
1. El corpus venía como `corpus_amazon.zip`; se descomprimió una **copia** en `datos/corpus_amazon/`. El zip y el resto de
   la carpeta original no se modificaron. Si la copia no existe, `utils.cargar_corpus()` lee directamente del zip.
2. Además de `text, rating, polarity, category, asin, verified_purchase` se conserva `id`, necesario para unir resultados
   entre notebooks y con la muestra de validación.
3. Los duplicados se detectaron comparando el texto en minúsculas y con espacios normalizados: resultaron **51** (no ~30;
   la diferencia viene de contar también variaciones de mayúsculas/espacios). Se conservó la primera aparición; 3 de esos
   textos aparecían con polaridades distintas.
4. Se eliminaron **123** reseñas de 3 palabras o menos. No había registros vacíos. Quedan **14,826** reseñas.

**Etiquetado**
5. El PDF pide etiquetado manual; por volumen se hizo **semiautomático con reglas** (`reglas_etiquetado.py`) más una muestra
   para validación manual (pendiente, ver sección 5).
6. Prioridad **Envío > Producto > Sentimiento**: Envío es la clase más rara y accionable, y los daños de transporte casi
   siempre usan también palabras de producto (*broken, cracked*); si Producto ganara, nunca se detectarían.
7. Se asumió que *"arrived/came broken, damaged, cracked, dented…"* es **daño de transporte (Envío)**, aunque en algunos
   casos el defecto pueda ser de fábrica. *"Missing parts/screws/pieces"* se tomó como **Producto** (kit incompleto);
   *"item missing from the package/order"*, *"wrong item"* y *"not what I ordered"* como **Envío** (problema de entrega).
8. Los elogios al envío (*fast shipping, arrived on time, well packaged*) se borran antes de evaluar el contexto, y las
   palabras de problema precedidas por una negación (hasta 5 palabras antes: *"no leaks"*, *"none arrived damaged"*) no cuentan.
9. Las reglas son deliberadamente conservadoras: algunas quejas vagas (*"junk"*, *"waste of money"*) quedan como Sentimiento
   porque no describen un problema concreto.

**Preprocesamiento y modelos**
10. Se usó **NLTK** (no spaCy): lematización WordNet con etiqueta gramatical, para reproducir exactamente el ejemplo del PDF
    (*"The package arrived late and damaged!!!" → "package arrive late damage"*). Las contracciones se expanden
    (*doesn't → does not*) para conservar la negación; se eliminan números; las letras repetidas se reducen a dos.
11. TF-IDF: unigramas + bigramas, `min_df=3`, `max_df=0.90`, `max_features=20000`, `sublinear_tf=True`, siempre dentro de un
    `Pipeline` (se ajusta solo con entrenamiento / con cada pliegue de entrenamiento). `rating` y `polarity` nunca son características.
12. SVM: en la misma malla compiten `LinearSVC` (6 valores de C) y `SVC` con kernel RBF (C = 1 y 10), ambos con
    `class_weight="balanced"`. El RBF quedó por debajo del lineal (F1 macro en CV 0.6624 vs. 0.7115) y cada ajuste tardó 97-155 s frente a 3-4 s del lineal.
13. KNN no admite pesos por clase; se buscó `weights` (uniform/distance). Naive Bayes usa `fit_prior=False`. RF y SVM usan
    `class_weight="balanced"`.
14. El SVM lineal no da probabilidades: la "confianza" de `clasificar_resena` es el **softmax de la función de decisión**
    (sirve para ordenar qué tan segura es la predicción, pero **no** es una probabilidad calibrada).
15. **Tiempos:** el clasificador usa un solo núcleo (RF con `n_jobs=1`); antes de medir se cierran los procesos auxiliares de
    `GridSearchCV` y se reporta la **mediana de 3 repeticiones**. En una primera corrida sin estas precauciones los tiempos de
    entrenamiento salían inflados (~7 s para todos los modelos) por competencia con esos procesos; por eso se corrigió y se
    volvieron a ejecutar los notebooks 04-09 (las métricas salieron idénticas). Aun así, los tiempos de pocos segundos varían entre
    ejecuciones según la carga del equipo: diferencias de ~1 s entre KNN, Naive Bayes y SVM no son significativas; la de Random
    Forest sí. Las métricas no dependen de la máquina (semilla 42 en todo).
16. Regla de selección fijada antes de ver resultados: mayor F1 macro en prueba; si otro modelo queda a < 0.005 (empate
    práctico), gana el de menor tiempo de entrenamiento y luego el de menor desviación en CV.
17. La muestra de validación está **estratificada por la etiqueta de las reglas** (no sigue la distribución natural) y viene
    solo de la partición de prueba, así que ningún modelo la vio al entrenar.
18. Las dependencias se instalaron en el Python 3.12 del sistema (sin entorno virtual). Versiones usadas: scikit-learn 1.9.1,
    pandas 3.1.1, numpy 2.5.5, nltk 3.10.4, matplotlib 3.11.2, seaborn 0.13.2, joblib 1.6.0.
