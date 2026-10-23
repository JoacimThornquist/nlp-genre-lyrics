# Clasificación de género musical a partir de letras de canciones

Trabajo práctico de Procesamiento de Lenguaje Natural — ITBA, 2026Q2.

**Integrantes:** Joacim Thörnquist · Alicia Cobo Iglesias · Santiago Minoyetti

## Objetivo

Predecir el género musical de una canción (pop, rap, rock, R&B, country) a partir de su letra y entender qué
información lingüística usa el clasificador: cuánto aporta la forma (estructura, repetición, rima, métrica) frente
al contenido, cuánto mejoran las representaciones contextuales y dónde fallan los modelos.

## Datos

"5 Million Song Lyrics" (Kaggle, v3): ~5,9 millones de letras de Genius con género, artista, año y vistas.
El archivo no está en el repo; ver [`data/README.md`](data/README.md) para descargarlo.

Muestra de trabajo: 10.000 canciones por género (se excluye `misc`), semilla 42 → 50.000 canciones de
33.401 artistas. Solo letras en inglés, sin encabezados de Genius, sin marcadores (`[Instrumental]`, "Coming soon"),
sin duplicados, sin cuentas de traducción de Genius y fuera de un bloque importado donde "pop" es la etiqueta por
defecto. Detalle en `notebooks/01_eda.ipynb`.

## Estructura

```
data/          datos (no versionados)
notebooks/     análisis numerados (01_eda, 02_rasgos, 03_modelos_rasgos, ...)
src/datos.py   lectura por partes del zip, conteos y limpieza de encabezados
src/muestra.py metadatos del dataset completo y construcción de la muestra
src/evaluacion.py  protocolo común: particiones por artista, predicciones fuera de muestra, métricas, registro
src/rasgos.py  rasgos de forma y estilo de cada letra (rima con el diccionario CMU, repetición, métrica, ...)
tests/         pruebas de src/rasgos.py
reports/       figuras y tablas que van al informe (registro de experimentos en reports/tablas)
docs/          consignas de las entregas
```

Todos los modelos usan las mismas cinco particiones de validación cruzada, estratificadas y agrupadas por artista
(`data/processed/folds.parquet`): ningún artista aparece a la vez en entrenamiento y evaluación. Cada modelo guarda
sus predicciones fuera de muestra en `data/processed/oof/` y una fila en `reports/tablas/registro_experimentos.csv`.
Las comparaciones usan macro-F1 con intervalos de confianza del 95% por bootstrap de artistas.

## Cómo reproducir

```bash
pip install -r requirements.txt
python -m src.datos          # prueba rápida de lectura del dataset
python -m src.muestra        # ~25 min: metadatos del total + muestra (data/processed/*.parquet)
python -m src.evaluacion     # ~3 min: particiones por artista y referencias M0 (mayoritaria) y M1 (TF-IDF + LR)
python -m src.rasgos         # ~15 min: 45 rasgos de forma y estilo por canción (rima, repetición, métrica, ...)
python tests/test_rasgos.py  # pruebas de los rasgos con estrofas anotadas a mano
```

Después, correr los notebooks de `notebooks/` en orden: `01_eda.ipynb` (~5 min), `02_rasgos.ipynb` (menos de un
minuto) y `03_modelos_rasgos.ipynb` (~45 min en CPU).
