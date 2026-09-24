# Datos

Esta carpeta no se versiona (salvo este archivo).

- `ds2.csv.zip`: dataset "5 Million Song Lyrics" (Kaggle, v3), letras de Genius. 3,3 GB comprimido,
  un único `ds2.csv` de 9,2 GB adentro. Columnas: `title, tag, artist, year, views, features, lyrics, id`.
  Descargarlo de Kaggle y dejarlo acá con ese nombre. **No descomprimirlo**: `src/datos.py` lee directo del zip.
- `processed/`: muestras y datasets derivados en parquet, generados por los notebooks.

En Colab (o si el zip está en otro lado), definir la variable de entorno `LYRICS_DATA` con la ruta al zip.
