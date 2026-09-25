"""Lectura del corpus de letras (Genius, Kaggle) sin cargarlo completo en memoria.

El CSV descomprimido ocupa alrededor de 9 GB, por lo que se lee directamente desde el archivo
comprimido y por partes. La ruta por defecto es data/ds2.csv.zip, relativa a la raíz del
repositorio; la variable de entorno LYRICS_DATA permite indicar otra ubicación.

Ejemplo de uso:
    from src.datos import leer_por_partes, contar_valores, limpiar_letra
    conteo = contar_valores("tag")          # un recorrido del archivo, solo esa columna
    for parte in leer_por_partes(columnas=["tag", "lyrics"]):
        ...
"""
from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_POR_DEFECTO = RAIZ / "data" / "ds2.csv.zip"

COLUMNAS = ["title", "tag", "artist", "year", "views", "features", "lyrics", "id"]
GENEROS = ["pop", "rap", "rock", "rb", "country"]  # se excluye "misc", que no es un género musical
SEMILLA = 42
FILAS_POR_PARTE = 200_000

_ENCABEZADO = re.compile(r"\[[^\]\n]*\]")
# Encabezados de sección sin corchetes que ocupan una línea completa: "(Chorus)", "{Hook}", "Verse 2:",
# "CHORUS", "Hook x2". La condición de línea completa evita eliminar versos que comienzan con la misma
# palabra ("Bridge over troubled water").
_SECCION = r"(?:verse|chorus|hook|intro|outro|bridge|pre-?chorus|post-?chorus|refrain|interlude|breakdown)"
_ENCABEZADO_SUELTO = re.compile(
    rf"^[ \t]*(?:[\(\{{<][ \t]*{_SECCION}\b[^\)\}}>\n]{{0,40}}[\)\}}>]"  # (Chorus X2 - Butch Cassidy)
    rf"|{_SECCION}[ \t]*(?:\d+|x[ \t]*\d+)?[ \t]*:[^\n]{{0,40}}"      # Verse 1: Kendrick Lamar
    rf"|{_SECCION}[ \t]*(?:\d+|x[ \t]*\d+)?)[ \t]*$",                 # CHORUS / Verse 2 / Hook x2
    re.IGNORECASE | re.MULTILINE,
)
_COMILLAS = {"’": "'", "‘": "'", "“": '"', "”": '"'}


def ruta_dataset(ruta: str | os.PathLike | None = None) -> Path:
    """Ruta al archivo comprimido, por orden de prioridad: argumento, LYRICS_DATA, data/ds2.csv.zip."""
    if ruta is not None:
        return Path(ruta)
    return Path(os.environ.get("LYRICS_DATA", RUTA_POR_DEFECTO))


def leer_por_partes(
    columnas: Iterable[str] | None = None,
    filas_por_parte: int = FILAS_POR_PARTE,
    ruta: str | os.PathLike | None = None,
    max_partes: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Recorre el CSV en DataFrames de `filas_por_parte` filas, leyendo solo las `columnas` indicadas."""
    lector = pd.read_csv(
        ruta_dataset(ruta),
        usecols=list(columnas) if columnas is not None else None,
        chunksize=filas_por_parte,
        compression="zip",
    )
    for i, parte in enumerate(lector):
        if max_partes is not None and i >= max_partes:
            break
        yield parte


def contar_valores(columna: str, **kwargs) -> pd.Series:
    """Frecuencia de cada valor de `columna` en el corpus completo, en un único recorrido."""
    total: Counter = Counter()
    for parte in leer_por_partes(columnas=[columna], **kwargs):
        total.update(parte[columna].value_counts(dropna=False).to_dict())
    return pd.Series(total, name=columna).sort_values(ascending=False)


def muestra_balanceada(
    n_por_genero: int = 10_000,
    generos: Iterable[str] = GENEROS,
    columnas: Iterable[str] = ("title", "tag", "artist", "year", "views", "lyrics", "id"),
    semilla: int = SEMILLA,
    conteos: pd.Series | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Muestra aleatoria simple de `n_por_genero` canciones por género, reproducible con `semilla`.

    Si no se proveen `conteos`, un primer recorrido cuenta las canciones por género. El segundo
    conserva cada fila con una probabilidad algo mayor que n/total de su género y luego recorta a
    exactamente n por género. Solo descarta letras vacías: no aplica los filtros de
    `src/muestra.py` (marcadores, duplicados, idioma, bloque de importación), por lo que la
    muestra de trabajo se construye con ese módulo.
    """
    generos = list(generos)
    columnas = list(dict.fromkeys(list(columnas) + ["tag", "lyrics"]))
    if conteos is None:
        conteos = contar_valores("tag", **kwargs)
    prob = {g: min(1.0, 1.3 * n_por_genero / conteos[g]) for g in generos}

    rng = np.random.default_rng(semilla)
    elegidas = []
    for parte in leer_por_partes(columnas=columnas, **kwargs):
        parte = parte[parte["tag"].isin(generos)]
        parte = parte[parte["lyrics"].notna() & (parte["lyrics"].str.strip() != "")]
        umbral = parte["tag"].map(prob).to_numpy()
        elegidas.append(parte[rng.random(len(parte)) < umbral])

    df = pd.concat(elegidas, ignore_index=True)
    # Una permutación aleatoria seguida de las primeras n filas de cada género equivale a un
    # muestreo sin reemplazo dentro de cada género.
    df = (
        df.sample(frac=1, random_state=semilla)
        .groupby("tag", observed=True)
        .head(n_por_genero)
        .reset_index(drop=True)
    )
    return df


def quitar_encabezados(letra: str) -> str:
    """Elimina los encabezados de sección de Genius, como [Chorus: Artista] o [Verse 1].

    Los encabezados contienen nombres de artistas y marcas de estructura que no forman parte de
    la letra. También se eliminan los mismos rótulos escritos sin corchetes en una línea propia
    ("(Chorus)", "Verse 2:", "HOOK"), presentes en el 12,8% de las canciones de rap y el 11,2%
    de las de R&B de la muestra.
    """
    sin = _ENCABEZADO.sub(" ", letra)
    sin = _ENCABEZADO_SUELTO.sub("", sin)
    return re.sub(r"[ \t]+", " ", sin).strip()


def limpiar_letra(letra: str) -> str:
    """Letra sin encabezados y con apóstrofos y comillas tipográficos reemplazados por los rectos.

    El uso del apóstrofo tipográfico aumenta con el tiempo: aparece en el 9% a 14% de las letras
    hasta 2009 y en el 26% a 37% de las posteriores a 2014. Sin normalizar, "don’t" y "don't"
    serían tokens distintos y el período quedaría codificado como un rasgo de formato.
    """
    for tipografica, recta in _COMILLAS.items():  # str.replace es más rápido que str.translate
        letra = letra.replace(tipografica, recta)
    return quitar_encabezados(letra)


if __name__ == "__main__":
    # Verificación de lectura: primera parte del archivo, tipos de columna y géneros presentes.
    primera = next(leer_por_partes(filas_por_parte=5_000))
    print(primera.shape)
    print(primera.dtypes)
    print(primera["tag"].value_counts())
    print(quitar_encabezados(primera.loc[0, "lyrics"])[:300])
