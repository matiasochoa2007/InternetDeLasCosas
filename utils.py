"""
utils.py
========
Funciones compartidas por todos los notebooks del proyecto
"Clasificación de reseñas de Amazon (Hogar y Cocina)".

Se agrupan aquí para que cada notebook sea corto y se concentre en explicar el
paso que le corresponde, y para garantizar que TODOS los modelos usen exactamente
el mismo preprocesamiento, la misma partición y las mismas métricas.
"""

import html
import json
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Rutas y constantes del proyecto (todas relativas a esta carpeta)
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATOS = BASE / "datos"
RESULTADOS = BASE / "resultados"
MODELOS = BASE / "modelos"
for _carpeta in (DATOS, RESULTADOS, MODELOS):
    _carpeta.mkdir(exist_ok=True)

CARPETA_CORPUS = DATOS / "corpus_amazon"           # corpus descomprimido (copia)
ZIP_CORPUS = BASE.parent / "corpus_amazon.zip"      # zip original (solo lectura)

ARCHIVO_LIMPIO = DATOS / "resenas_limpias.csv"
ARCHIVO_ETIQUETADO = DATOS / "resenas_etiquetadas.csv"
ARCHIVO_PROCESADO = DATOS / "resenas_procesadas.csv"
ARCHIVO_VALIDACION = DATOS / "muestra_validacion.csv"

SEMILLA = 42
CLASES = ["Envío", "Producto", "Sentimiento"]
N_PLIEGUES = 5

# Nombre "bonito" de cada modelo -> nombre de carpeta/archivo
MODELOS_PROYECTO = {
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "SVM": "svm",
    "Random Forest": "random_forest",
}


# ---------------------------------------------------------------------------
# 2. Carga del corpus
# ---------------------------------------------------------------------------
COLUMNAS_CONSERVAR = ["id", "text", "rating", "polarity", "category", "asin", "verified_purchase"]


def cargar_corpus():
    """
    Lee los 15,000 JSON (uno por reseña) y devuelve un DataFrame.
    Usa la carpeta descomprimida datos/corpus_amazon/; si no existe, lee
    directamente del zip original sin modificarlo.
    """
    registros = []
    if CARPETA_CORPUS.exists():
        for archivo in sorted(CARPETA_CORPUS.glob("*/*.json")):
            with open(archivo, encoding="utf-8") as f:
                d = json.load(f)
            d["carpeta"] = archivo.parent.name
            registros.append(d)
    elif ZIP_CORPUS.exists():
        with zipfile.ZipFile(ZIP_CORPUS) as z:
            for nombre in sorted(z.namelist()):
                if nombre.endswith(".json"):
                    d = json.loads(z.read(nombre).decode("utf-8"))
                    d["carpeta"] = Path(nombre).parent.name
                    registros.append(d)
    else:
        raise FileNotFoundError("No se encontró datos/corpus_amazon/ ni corpus_amazon.zip")
    df = pd.DataFrame(registros)
    # product_metadata es un diccionario anidado que no se usa en el modelo
    return df.drop(columns=["product_metadata"], errors="ignore")


# ---------------------------------------------------------------------------
# 3. Preprocesamiento de texto (en inglés)
# ---------------------------------------------------------------------------
# Negaciones que se CONSERVAN aunque estén en la lista de stopwords de NLTK,
# porque cambian el significado ("not working" != "working").
NEGACIONES_CONSERVAR = {"not", "no", "nor", "never", "none", "nobody", "nothing",
                        "neither", "nowhere", "without"}

_RECURSOS_NLTK = {
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
}
_STOPWORDS = None
_LEMATIZADOR = None


def preparar_nltk():
    """Descarga (solo la primera vez) los recursos de NLTK que se necesitan."""
    import nltk
    global _STOPWORDS, _LEMATIZADOR
    for ruta, nombre in _RECURSOS_NLTK.items():
        try:
            nltk.data.find(ruta)
        except LookupError:
            nltk.download(nombre, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _STOPWORDS = set(stopwords.words("english")) - NEGACIONES_CONSERVAR
    _LEMATIZADOR = WordNetLemmatizer()


def _etiqueta_wordnet(etiqueta_pos):
    """Convierte la etiqueta gramatical de NLTK (Penn Treebank) al formato de WordNet."""
    from nltk.corpus import wordnet
    if etiqueta_pos.startswith("J"):
        return wordnet.ADJ
    if etiqueta_pos.startswith("V"):
        return wordnet.VERB
    if etiqueta_pos.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def limpiar_texto(texto):
    """
    Limpieza "superficial" (sin tokenizar):
    Unicode -> minúsculas -> sin HTML/URLs -> negaciones expandidas ->
    sin puntuación/números/caracteres especiales -> letras repetidas reducidas ->
    espacios corregidos.
    """
    t = html.unescape(str(texto))                              # &amp; -> &, &#34; -> "
    t = unicodedata.normalize("NFKD", t)                       # normalización Unicode
    t = t.encode("ascii", "ignore").decode("ascii")            # quita acentos y emojis
    t = t.lower()                                              # minúsculas
    t = re.sub(r"<[^>]+>", " ", t)                             # etiquetas HTML (<br />)
    t = re.sub(r"\[\[asin:[^\]]*\]\]", " ", t)                 # enlaces internos de Amazon
    t = re.sub(r"(https?://\S+|www\.\S+)", " ", t)             # URLs
    t = t.replace("’", "'")
    # Negaciones: "won't" -> "will not", "can't" -> "can not", "doesn't" -> "does not"
    t = re.sub(r"\bwon't\b", "will not", t)
    t = re.sub(r"\bcan't\b", "can not", t)
    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"[^a-z\s]", " ", t)                            # puntuación, números, símbolos
    t = re.sub(r"([a-z])\1{2,}", r"\1\1", t)                   # "soooo" -> "soo"
    t = re.sub(r"\s+", " ", t).strip()                         # espacios redundantes
    return t


def preprocesar_texto(texto):
    """
    Preprocesamiento completo de una reseña:
    limpieza -> tokenización -> etiquetado gramatical -> lematización ->
    eliminación de stopwords (conservando negaciones) -> texto final.

    Ejemplo del protocolo:
        "The package arrived late and damaged!!!" -> "package arrive late damage"
    """
    from nltk import pos_tag, word_tokenize
    if _STOPWORDS is None:
        preparar_nltk()
    limpio = limpiar_texto(texto)
    tokens = word_tokenize(limpio)
    etiquetados = pos_tag(tokens)
    lemas = [_LEMATIZADOR.lemmatize(tok, _etiqueta_wordnet(pos)) for tok, pos in etiquetados]
    finales = [l for l in lemas if l not in _STOPWORDS and len(l) > 1]
    return " ".join(finales)


# ---------------------------------------------------------------------------
# 4. Vectorización y validación cruzada (idénticas para los 4 modelos)
# ---------------------------------------------------------------------------
PARAMETROS_TFIDF = dict(
    ngram_range=(1, 2),     # unigramas y bigramas ("not work", "arrive late")
    min_df=3,               # el término debe aparecer en al menos 3 reseñas
    max_df=0.90,            # ignora términos presentes en más del 90 % de reseñas
    max_features=20000,     # vocabulario acotado
    sublinear_tf=True,      # 1 + log(tf): reduce el peso de palabras muy repetidas
)


def crear_vectorizador():
    """TF-IDF con los mismos parámetros para todos los modelos."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(**PARAMETROS_TFIDF)


def crear_cv():
    """Validación cruzada estratificada de 5 pliegues con semilla fija."""
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=SEMILLA)


def cargar_particiones():
    """Lee el CSV procesado y devuelve (df, entrenamiento, prueba) según la columna 'particion'."""
    df = pd.read_csv(ARCHIVO_PROCESADO, encoding="utf-8-sig", keep_default_na=False)
    entrenamiento = df[df["particion"] == "entrenamiento"].reset_index(drop=True)
    prueba = df[df["particion"] == "prueba"].reset_index(drop=True)
    return df, entrenamiento, prueba


def buscar_hiperparametros(pipeline, malla, X, y, n_jobs=-1):
    """
    GridSearchCV SOLO sobre entrenamiento, con F1 macro como métrica principal
    (también registra accuracy). El TF-IDF está dentro del Pipeline, así que en
    cada pliegue se ajusta únicamente con los datos de ese pliegue (sin fuga).
    """
    from sklearn.model_selection import GridSearchCV
    busqueda = GridSearchCV(
        pipeline, malla,
        scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
        refit="f1_macro", cv=crear_cv(), n_jobs=n_jobs, return_train_score=False,
    )
    inicio = time.perf_counter()
    busqueda.fit(X, y)
    duracion = time.perf_counter() - inicio
    return busqueda, duracion


def tabla_busqueda(busqueda, n=10):
    """Tabla ordenada con las mejores combinaciones de hiperparámetros."""
    r = pd.DataFrame(busqueda.cv_results_)
    columnas = ["params", "mean_test_f1_macro", "std_test_f1_macro",
                "mean_test_accuracy", "std_test_accuracy", "mean_fit_time", "rank_test_f1_macro"]
    r = r[columnas].sort_values("rank_test_f1_macro")
    r["params"] = r["params"].astype(str)
    return r.head(n).reset_index(drop=True), r


def resumen_cv(busqueda):
    """Media y desviación estándar de la validación cruzada para la mejor combinación."""
    i = busqueda.best_index_
    r = busqueda.cv_results_
    return {
        "cv_f1_macro_media": float(r["mean_test_f1_macro"][i]),
        "cv_f1_macro_std": float(r["std_test_f1_macro"][i]),
        "cv_accuracy_media": float(r["mean_test_accuracy"][i]),
        "cv_accuracy_std": float(r["std_test_accuracy"][i]),
    }


def medir_entrenamiento_y_prediccion(modelo, X_train, y_train, X_test, repeticiones=3):
    """
    Reentrena el pipeline y mide (con time.perf_counter) el tiempo de entrenamiento y de predicción.

    Para que la medición sea justa entre modelos:
    - antes de medir se cierran los procesos auxiliares que dejó GridSearchCV (si siguen vivos
      compiten por el procesador e inflan los tiempos);
    - se repite 'repeticiones' veces y se reporta la MEDIANA.
    El modelo es determinista (semilla fija), así que todas las repeticiones dan las mismas predicciones.
    """
    from sklearn.base import clone
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass
    tiempos_ent, tiempos_pred = [], []
    for _ in range(repeticiones):
        candidato = clone(modelo)
        inicio = time.perf_counter()
        candidato.fit(X_train, y_train)
        tiempos_ent.append(time.perf_counter() - inicio)
        inicio = time.perf_counter()
        y_pred = candidato.predict(X_test)
        tiempos_pred.append(time.perf_counter() - inicio)
    return candidato, y_pred, float(np.median(tiempos_ent)), float(np.median(tiempos_pred))


# ---------------------------------------------------------------------------
# 5. Métricas, matriz de confusión y guardado de resultados
# ---------------------------------------------------------------------------
def calcular_metricas(y_true, y_pred):
    """Accuracy, precisión, recall y F1 (macro, ponderado y por clase)."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    m = {"accuracy": accuracy_score(y_true, y_pred)}
    for promedio, sufijo in (("macro", "macro"), ("weighted", "ponderado")):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=promedio, zero_division=0)
        m[f"precision_{sufijo}"], m[f"recall_{sufijo}"], m[f"f1_{sufijo}"] = p, r, f
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASES, zero_division=0)
    for i, c in enumerate(CLASES):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"], m[f"soporte_{c}"] = p[i], r[i], f[i], int(s[i])
    m["n_prueba"] = int(len(y_true))
    m["n_errores"] = int((np.asarray(y_true) != np.asarray(y_pred)).sum())
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def graficar_matriz_confusion(y_true, y_pred, titulo, carpeta, ax=None, guardar=True):
    """Dibuja la matriz de confusión (conteos) y la guarda en PNG y CSV."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    matriz = confusion_matrix(y_true, y_pred, labels=CLASES)
    tabla = pd.DataFrame(matriz, index=[f"real_{c}" for c in CLASES],
                         columns=[f"pred_{c}" for c in CLASES])
    propio = ax is None
    if propio:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASES, yticklabels=CLASES, ax=ax)
    ax.set_xlabel("Etiqueta predicha")
    ax.set_ylabel("Etiqueta real (reglas)")
    ax.set_title(titulo)
    if guardar:
        carpeta = Path(carpeta)
        tabla.to_csv(carpeta / "matriz_confusion.csv", encoding="utf-8-sig")
        if propio:
            fig.tight_layout()
            fig.savefig(carpeta / "matriz_confusion.png", dpi=120)
    return tabla


def puntajes_confianza(modelo, textos):
    """
    Devuelve una matriz (n_textos x n_clases) con la confianza de cada clase.
    - Si el modelo tiene predict_proba (KNN, Naive Bayes, Random Forest) se usan probabilidades.
    - Si no (SVM lineal), se aplica softmax a la función de decisión: es una
      puntuación de confianza relativa, NO una probabilidad calibrada.
    """
    if hasattr(modelo, "predict_proba"):
        try:
            return modelo.predict_proba(textos), "probabilidad"
        except AttributeError:
            pass
    d = np.atleast_2d(modelo.decision_function(textos))
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True), "softmax de la función de decisión (no calibrada)"


def guardar_predicciones(prueba, y_pred, modelo, carpeta):
    """Guarda predicciones_prueba.csv con el texto, la etiqueta real, la predicha y la confianza."""
    tabla = prueba[["id", "text", "rating", "polarity", "etiqueta_reglas"]].copy()
    tabla = tabla.rename(columns={"etiqueta_reglas": "etiqueta_real"})
    tabla["etiqueta_predicha"] = y_pred
    tabla["correcta"] = tabla["etiqueta_real"] == tabla["etiqueta_predicha"]
    conf, _ = puntajes_confianza(modelo, prueba["texto_procesado"])
    for i, c in enumerate(modelo.classes_):
        tabla[f"confianza_{c}"] = np.round(conf[:, i], 4)
    tabla.to_csv(Path(carpeta) / "predicciones_prueba.csv", index=False, encoding="utf-8-sig")
    return tabla


def guardar_metricas(nombre, metricas, carpeta):
    """Guarda metricas.json (todas las cifras del modelo) en su carpeta de resultados."""
    metricas = {"modelo": nombre, **metricas}
    with open(Path(carpeta) / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    return metricas


def guardar_reporte(nombre, y_true, y_pred, metricas, carpeta):
    """classification_report de scikit-learn + resumen de tiempos y CV en texto plano."""
    from sklearn.metrics import classification_report
    reporte = classification_report(y_true, y_pred, labels=CLASES, digits=4, zero_division=0)
    texto = (
        f"Modelo: {nombre}\n"
        f"Exactitud (accuracy) en prueba: {metricas['accuracy']:.4f}\n"
        f"F1 macro en prueba: {metricas['f1_macro']:.4f}\n"
        f"F1 macro en validación cruzada (5 pliegues): {metricas['cv_f1_macro_media']:.4f} "
        f"± {metricas['cv_f1_macro_std']:.4f}\n"
        f"Tiempo de entrenamiento: {metricas['tiempo_entrenamiento_s']:.3f} s | "
        f"Tiempo de predicción ({metricas['n_prueba']} reseñas): {metricas['tiempo_prediccion_s']:.3f} s\n\n"
        f"{reporte}"
    )
    (Path(carpeta) / "reporte_clasificacion.txt").write_text(texto, encoding="utf-8")
    return reporte


# ---------------------------------------------------------------------------
# 6. Análisis de errores
# ---------------------------------------------------------------------------
def ejemplos_errores(predicciones, carpeta, n_por_tipo=3, max_caracteres=300):
    """
    Agrupa las reseñas mal clasificadas por tipo de confusión (real -> predicha),
    guarda errores_por_tipo.csv y ejemplos_errores.csv, y devuelve ambos.
    """
    errores = predicciones[~predicciones["correcta"]].copy()
    conteo = (errores.groupby(["etiqueta_real", "etiqueta_predicha"]).size()
              .sort_values(ascending=False).rename("n_errores").reset_index())
    # Se revuelven los errores (semilla fija) y se toman n por tipo de confusión,
    # ordenando los tipos de mayor a menor frecuencia.
    revueltos = errores.sort_values("id").sample(frac=1, random_state=SEMILLA)
    ejemplos = revueltos.groupby(["etiqueta_real", "etiqueta_predicha"]).head(n_por_tipo)
    orden = {(r.etiqueta_real, r.etiqueta_predicha): i for i, r in conteo.iterrows()}
    ejemplos = ejemplos.assign(_orden=[orden[(a, b)] for a, b in
                                       zip(ejemplos["etiqueta_real"], ejemplos["etiqueta_predicha"])])
    ejemplos = ejemplos.sort_values("_orden")[["etiqueta_real", "etiqueta_predicha", "rating", "text"]].copy()
    ejemplos["text"] = ejemplos["text"].str.replace(r"<[^>]+>", " ", regex=True).str.slice(0, max_caracteres)
    conteo.to_csv(Path(carpeta) / "errores_por_tipo.csv", index=False, encoding="utf-8-sig")
    ejemplos.to_csv(Path(carpeta) / "ejemplos_errores.csv", index=False, encoding="utf-8-sig")
    return conteo, ejemplos


def desempeno_por_polaridad(predicciones, carpeta):
    """¿Se clasifican peor las reseñas de 3 estrellas (neutral)? Accuracy y F1 macro por polaridad."""
    from sklearn.metrics import accuracy_score, f1_score
    filas = []
    for polaridad, g in predicciones.groupby("polarity"):
        filas.append({
            "polaridad": polaridad,
            "n": len(g),
            "accuracy": accuracy_score(g["etiqueta_real"], g["etiqueta_predicha"]),
            "f1_macro": f1_score(g["etiqueta_real"], g["etiqueta_predicha"], labels=CLASES,
                                 average="macro", zero_division=0),
        })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(Path(carpeta) / "desempeno_por_polaridad.csv", index=False, encoding="utf-8-sig")
    return tabla


def palabras_influyentes_lineales(pesos, vocabulario, clases, n=15):
    """Top-n términos con mayor peso por clase (coeficientes de SVM o log-probabilidades de NB)."""
    tabla = {}
    for i, c in enumerate(clases):
        orden = np.argsort(pesos[i])[::-1][:n]
        tabla[c] = [f"{vocabulario[j]} ({pesos[i][j]:.2f})" for j in orden]
    return pd.DataFrame(tabla)


# ---------------------------------------------------------------------------
# 7. Clasificar una reseña nueva con el mejor modelo guardado
# ---------------------------------------------------------------------------
_MEJOR_MODELO = None


def clasificar_resena(texto, ruta_modelo=None):
    """
    Carga (una sola vez) el mejor modelo guardado en modelos/mejor_modelo.joblib,
    preprocesa el texto igual que en el entrenamiento y devuelve:
        {"etiqueta": ..., "confianza": ..., "puntajes": {clase: valor}, "tipo_confianza": ...}
    """
    import joblib
    global _MEJOR_MODELO
    if ruta_modelo is not None:
        modelo = joblib.load(ruta_modelo)
    else:
        if _MEJOR_MODELO is None:
            _MEJOR_MODELO = joblib.load(MODELOS / "mejor_modelo.joblib")
        modelo = _MEJOR_MODELO
    procesado = preprocesar_texto(texto)
    etiqueta = modelo.predict([procesado])[0]
    conf, tipo = puntajes_confianza(modelo, [procesado])
    puntajes = {c: round(float(p), 4) for c, p in zip(modelo.classes_, conf[0])}
    return {"etiqueta": etiqueta, "confianza": puntajes[etiqueta],
            "puntajes": puntajes, "tipo_confianza": tipo}


# ---------------------------------------------------------------------------
# 8. Validación contra la muestra etiquetada a mano
# ---------------------------------------------------------------------------
_EQUIVALENCIAS = {
    "envio": "Envío", "envío": "Envío", "e": "Envío", "shipping": "Envío",
    "producto": "Producto", "p": "Producto", "product": "Producto",
    "sentimiento": "Sentimiento", "s": "Sentimiento", "sentiment": "Sentimiento",
}


def normalizar_etiqueta_manual(valor):
    """Acepta 'Envío', 'envio', 'E', 'p', 'Sentimiento', etc. Devuelve la etiqueta oficial o None."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return None
    v = str(valor).strip().lower()
    return _EQUIVALENCIAS.get(v) if v else None
