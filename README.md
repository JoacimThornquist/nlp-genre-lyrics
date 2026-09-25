# Clasificación de género musical a partir de letras de canciones

Trabajo práctico de Procesamiento de Lenguaje Natural — ITBA, 2026Q2.

**Integrantes:** Joacim Thörnquist · Alicia Cobo Iglesias · Santiago Minoyetti

## Objetivo

Predecir el género musical de una canción (pop, rap, rock, R&B, country) a partir de su letra y analizar qué
rasgos del lenguaje distinguen a cada género. *(Borrador: se precisa en la primera entrega.)*

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
notebooks/     análisis numerados (01_eda, 02_baselines, ...)
src/datos.py   lectura por partes del zip, conteos y limpieza de encabezados
src/muestra.py metadatos del dataset completo y construcción de la muestra
reports/       figuras y tablas que van al informe
docs/          consignas de las entregas
```

## Cómo reproducir

```bash
pip install -r requirements.txt
python -m src.datos          # prueba rápida de lectura del dataset
python -m src.muestra        # ~25 min: metadatos del total + muestra (data/processed/*.parquet)
```

Después, correr los notebooks de `notebooks/` en orden (`01_eda.ipynb` tarda ~5 min).
