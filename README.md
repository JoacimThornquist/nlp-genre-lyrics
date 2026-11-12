# Clasificación de género musical a partir de letras de canciones

Trabajo práctico de Procesamiento de Lenguaje Natural — ITBA, 2026Q2.

**Integrantes:** Joacim Thörnquist · Alicia Cobo Iglesias · Santiago Minoyetti

## Objetivo

Predecir el género musical de una canción (pop, rap, rock, R&B, country) a partir de su letra y entender qué
información lingüística usa el clasificador. Tres preguntas:

1. **Forma y contenido.** ¿Cuánto aporta la forma de la letra (estructura, repetición, rima, métrica) frente al
   contenido (qué palabras usa)?
2. **Representaciones contextuales.** ¿Cuánto mejora un transformer respecto de un modelo de palabras, y cuánto
   aporta ajustarlo a la tarea?
3. **Errores.** ¿Dónde fallan los modelos? ¿El pop es un género con lenguaje propio o una etiqueta residual?

## Datos

"5 Million Song Lyrics" (Kaggle, v3): ~5,9 millones de letras de Genius con género, artista, año y vistas.
El archivo no está en el repo; ver [`data/README.md`](data/README.md) para descargarlo.

Muestra de trabajo: 10.000 canciones por género (se excluye `misc`), semilla 42 → 50.000 canciones de
33.401 artistas. Solo letras en inglés, sin encabezados de Genius, sin marcadores (`[Instrumental]`, "Coming soon"),
sin duplicados, sin cuentas de traducción de Genius y fuera de un bloque importado donde "pop" es la etiqueta por
defecto. Detalle en `notebooks/01_eda.ipynb`.

## Resultados

Macro-F1 con cinco particiones agrupadas por artista; IC 95% por bootstrap de artistas (1.000 remuestreos).

| Modelo | macro-F1 | IC 95% | Pop | Rap | Rock | R&B | Country | Δ vs. TF-IDF [IC 95%] |
|---|---|---|---|---|---|---|---|---|
| Clase mayoritaria | 0,067 | [0,064; 0,070] | 0,000 | 0,000 | 0,000 | 0,000 | 0,333 | −0,534 [−0,538; −0,529] |
| Forma pura (24 rasgos) + XGBoost | 0,495 | [0,490; 0,500] | 0,316 | 0,704 | 0,448 | 0,444 | 0,562 | −0,106 [−0,111; −0,101] |
| Rasgos de forma y estilo (41) + XGBoost | 0,549 | [0,544; 0,554] | 0,347 | 0,777 | 0,510 | 0,497 | 0,614 | −0,051 [−0,056; −0,047] |
| MiniLM congelado + regresión logística | 0,558 | [0,553; 0,563] | 0,346 | 0,743 | 0,551 | 0,518 | 0,633 | −0,042 [−0,047; −0,038] |
| **TF-IDF + regresión logística** | 0,600 | [0,596; 0,605] | 0,385 | 0,797 | 0,573 | 0,564 | 0,683 | — |
| TF-IDF + rasgos, misma regresión | 0,617 | [0,612; 0,621] | 0,410 | 0,807 | 0,587 | 0,576 | 0,701 | +0,016 [0,013; 0,019] |
| MiniLM con ajuste fino | 0,604 | [0,599; 0,610] | 0,399 | 0,784 | 0,578 | 0,560 | 0,700 | +0,004 [0,000; 0,008] |

- **Forma y contenido.** Sumar los 41 rasgos a TF-IDF en la misma regresión logística mejora el macro-F1 en
  +0,016 (IC 95% [0,013; 0,019]), con mejora en los cinco géneros. Con 24 rasgos de forma pura, sin ninguna
  palabra, se llega a 0,495.
- **Representaciones contextuales.** El mismo encoder (`all-MiniLM-L6-v2`, primeros 256 tokens de cada letra) da
  0,558 congelado, con una regresión logística encima, y 0,604 con ajuste fino: el ajuste aporta +0,046 (IC 95%
  [0,042; 0,050]). Congelado queda por debajo de TF-IDF (−0,042); con ajuste fino lo iguala (+0,004, con un
  intervalo que incluye el 0).
- **El pop.** Solo el 37% del pop se clasifica como pop. Al entrenar sin pop, el recall del rock pasa de 60% a 72% y
  el del R&B de 54% a 64%, y las canciones de pop se reparten entre rock (42%), R&B (33%), country (20%) y rap
  (6%). En tópicos LDA, el pop es el género más cercano al resto (Jensen-Shannon 0,127 frente a 0,185–0,348). De las 20
  canciones más parecidas a una canción de pop (embeddings de MiniLM), solo el 23% es de pop (azar: 20%; rap: 66%).

Todas las tablas están en `reports/tablas/` (resumen en `resultados_modelos.csv` y `resultados.json`) y las
figuras en `reports/figures/`.

## Estructura

```
data/                  datos (no versionados)
notebooks/
  01_eda               exploración del corpus, filtros y controles de fuga
  02_rasgos            descripción de los rasgos de forma por género
  03_modelos_rasgos    XGBoost sobre rasgos: ablación, permutación, SHAP y modelo apilado
  04_embeddings_minilm MiniLM congelado y con ajuste fino (GPU; pensado para Colab)
  06_pop_residual      tópicos LDA y distancia de Jensen-Shannon entre géneros
src/
  datos.py             lectura por partes del zip, conteos y limpieza de encabezados
  muestra.py           metadatos del dataset completo y construcción de la muestra
  evaluacion.py        protocolo común: particiones por artista, predicciones fuera de muestra, métricas, registro
  rasgos.py            41 rasgos de forma y estilo de cada letra, más 4 de transcripción como control
  modelos_texto.py     TF-IDF + regresión logística: elección de C, texto + rasgos, modelo sin pop
  resultados.py        comparación final de modelos, matriz de confusión, sin pop, pureza de vecinos, figuras
tests/                 pruebas de src/rasgos.py
reports/               figuras y tablas (registro de experimentos en reports/tablas)
docs/                  consignas de las entregas
```

## Protocolo de evaluación

- Todos los modelos usan las mismas cinco particiones, estratificadas y agrupadas por artista
  (`data/processed/folds.parquet`): ningún artista aparece a la vez en entrenamiento y evaluación.
- Lo que haya que elegir (C de la regresión, cantidad de árboles, época del ajuste fino) se decide con una
  validación interna: el 15% de los artistas del entrenamiento de cada partición. La partición de evaluación no
  interviene. XGBoost y el ajuste fino usan valores estándar fijados de antemano.
- Cada modelo guarda sus predicciones fuera de muestra en `data/processed/oof/` y una fila en
  `reports/tablas/registro_experimentos.csv`.
- Las comparaciones entre modelos son apareadas: mismas canciones y mismos remuestreos de artistas. Una
  diferencia se considera real si el IC 95% de la diferencia no incluye el 0.

## Cómo reproducir

```bash
pip install -r requirements.txt
python -m src.datos            # prueba rápida de lectura del dataset
python -m src.muestra          # ~25 min: metadatos del total + muestra (data/processed/*.parquet)
python -m src.evaluacion       # ~3 min: particiones por artista y referencias
python -m src.rasgos           # ~15 min: rasgos de forma y estilo por canción
python tests/test_rasgos.py    # pruebas de los rasgos con estrofas anotadas a mano
python -m src.modelos_texto    # ~20 min: elección de C, texto + rasgos, modelo sin pop, forma pura
```

Notebooks, en orden: `01_eda` (~5 min), `02_rasgos` (<1 min), `03_modelos_rasgos` (~45 min en CPU) y
`06_pop_residual` (~16 min).

MiniLM necesita GPU. `python -m src.resultados --exportar` genera `data/processed/letras_colab.parquet`; el
notebook `04_embeddings_minilm` lo recibe (en Colab, con GPU T4) y devuelve `salida_minilm.zip`, que se copia a
`data/processed/`.

```bash
python -m src.resultados       # ~2 min: tablas y figuras finales (usa salida_minilm.zip si está)
```
