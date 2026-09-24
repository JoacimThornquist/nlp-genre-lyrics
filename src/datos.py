"""Lectura del dataset de letras (Genius / Kaggle) sin cargarlo entero en memoria.

El CSV descomprimido pesa ~9 GB, así que todo se lee directo del zip y por partes.
Ruta por defecto: data/ds2.csv.zip (relativa a la raíz del repo). En Colab u otra
máquina se puede apuntar a otro lugar con la variable de entorno LYRICS_DATA.

Uso rápido:
    from src.datos import leer_por_partes, contar_valores, muestra_balanceada
    conteo = contar_valores("tag")                       # una pasada, solo esa columna
    df = muestra_balanceada(n_por_genero=10_000)         # dos pasadas
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
GENEROS = ["pop", "rap", "rock", "rb", "country"]  # "misc" excluido: no son canciones
SEMILLA = 42
FILAS_POR_PARTE = 200_000

_ENCABEZADO = re.compile(r"\[[^\]\n]*\]")


def ruta_dataset(ruta: str | os.PathLike | None = None) -> Path:
    """Ruta al zip: argumento explícito > variable LYRICS_DATA > data/ds2.csv.zip."""
    if ruta is not None:
        return Path(ruta)
    return Path(os.environ.get("LYRICS_DATA", RUTA_POR_DEFECTO))


def leer_por_partes(
    columnas: Iterable[str] | None = None,
    filas_por_parte: int = FILAS_POR_PARTE,
    ruta: str | os.PathLike | None = None,
    max_partes: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Itera el CSV en DataFrames de `filas_por_parte` filas, leyendo solo `columnas`."""
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
    """Frecuencia de cada valor de `columna` sobre el dataset completo (una pasada)."""
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
    """Muestra aleatoria de `n_por_genero` canciones por género, reproducible con `semilla`.

    Primera pasada (si no se pasan `conteos`): cuenta canciones por género.
    Segunda pasada: se queda con cada fila con probabilidad algo mayor a n/total
    de su género y al final recorta exactamente a n por género.
    Quita letras vacías; NO deduplica ni filtra idioma (eso va en el preprocesamiento).

    Nota: si la muestra canónica del grupo se generó en Colab con otro método,
    usar ese código para que el informe y el repo coincidan.
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
    # Mezclar y quedarse con las primeras n de cada género = muestreo sin reemplazo por género.
    df = (
        df.sample(frac=1, random_state=semilla)
        .groupby("tag", observed=True)
        .head(n_por_genero)
        .reset_index(drop=True)
    )
    return df


def quitar_encabezados(letra: str) -> str:
    """Quita los encabezados de sección de Genius: [Chorus: Artista], [Verse 1], etc.

    Esos corchetes traen nombres de artistas (fuga) y marcas de estructura que no son letra.
    """
    sin = _ENCABEZADO.sub(" ", letra)
    return re.sub(r"[ \t]+", " ", sin).strip()


if __name__ == "__main__":
    # Prueba rápida: lee la primera parte y muestra forma, columnas y géneros.
    primera = next(leer_por_partes(filas_por_parte=5_000))
    print(primera.shape)
    print(primera.dtypes)
    print(primera["tag"].value_counts())
    print(quitar_encabezados(primera.loc[0, "lyrics"])[:300])
