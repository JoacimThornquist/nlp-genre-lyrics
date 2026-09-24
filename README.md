# Clasificación de género musical a partir de letras de canciones

Trabajo práctico de Procesamiento de Lenguaje Natural — ITBA, 2026Q2.

**Integrantes:** Joacim Thörnquist · Alicia Cobo Iglesias · Santiago Minoyetti

## Objetivo

Predecir el género musical de una canción (pop, rap, rock, R&B, country) a partir de su letra y analizar qué
rasgos del lenguaje distinguen a cada género. *(Borrador: se precisa en la primera entrega.)*

## Datos

"5 Million Song Lyrics" (Kaggle, v3): ~5,9 millones de letras de Genius con género, artista, año y vistas.
El archivo no está en el repo; ver [`data/README.md`](data/README.md) para descargarlo.

Muestra de trabajo: 10.000 canciones por género (se excluye `misc`), semilla 42, sin letras vacías ni
duplicadas → 49.720 canciones de 10.832 artistas.

## Estructura

```
data/          datos (no versionados)
notebooks/     análisis numerados (01_eda, 02_baselines, ...)
src/datos.py   lectura por partes del zip, conteos, muestreo y limpieza
reports/       figuras que van al informe
docs/          consignas de las entregas
```

## Cómo reproducir

```bash
pip install -r requirements.txt
python -m src.datos          # prueba rápida de lectura del dataset
```

Después, correr los notebooks de `notebooks/` en orden.
