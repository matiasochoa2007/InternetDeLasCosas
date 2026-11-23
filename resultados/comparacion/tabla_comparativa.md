# Tabla comparativa de modelos (conjunto de prueba, n = 2966 reseñas)

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (ponderado) | F1 Envío | F1 Producto | F1 Sentimiento | F1 macro CV (media) | F1 macro CV (desv. est.) | Brecha CV - prueba | Errores en prueba | Tiempo búsqueda (s) | Tiempo entrenamiento (s) | Tiempo predicción (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KNN | 0.7063 | 0.6598 | 0.5264 | 0.5586 | 0.6912 | 0.3191 | 0.5723 | 0.7845 | 0.5720 | 0.0270 | 0.0134 | 871 | 44.8065 | 1.1738 | 0.7097 |
| Naive Bayes | 0.7202 | 0.5940 | 0.6123 | 0.5955 | 0.7237 | 0.3310 | 0.7024 | 0.7532 | 0.6248 | 0.0094 | 0.0292 | 830 | 17.6693 | 2.1131 | 0.1955 |
| SVM | 0.8240 | 0.7681 | 0.6955 | 0.7241 | 0.8225 | 0.5310 | 0.7782 | 0.8631 | 0.7115 | 0.0126 | -0.0126 | 522 | 192.9144 | 2.3042 | 0.1992 |
| Random Forest | 0.7839 | 0.6762 | 0.6860 | 0.6807 | 0.7852 | 0.4748 | 0.7399 | 0.8272 | 0.6973 | 0.0234 | 0.0166 | 641 | 227.8205 | 15.7856 | 0.7972 |

Métricas de prueba con etiquetas generadas por reglas. CV = validación cruzada estratificada de 5 pliegues sobre entrenamiento. Brecha = F1 macro CV − F1 macro prueba. Tiempos medidos con time.perf_counter().
