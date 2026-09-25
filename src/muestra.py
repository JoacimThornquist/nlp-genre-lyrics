"""Construcción de la muestra de trabajo a partir del corpus completo.

El proceso consta de tres pasos; cada uno guarda su resultado en data/processed/
(el análisis correspondiente está en notebooks/01_eda.ipynb):

1. `construir_metadatos()`: un recorrido del archivo comprimido (alrededor de 10 minutos) que
   genera metadatos.parquet, con una fila por canción (5,9 millones) y sin la letra: etiqueta,
   artista, año, vistas, longitud, caracteres no ASCII, cantidad de encabezados y un hash de la
   letra. Permite analizar el corpus completo sin volver a leer el archivo.
2. `elegir_candidatas()` y `extraer_letras()`: filtrado sobre los metadatos (género, bloque de
   importación, textos repetidos), sorteo de `n` canciones por género y un segundo recorrido que
   recupera solo esas letras.
3. `preparar_muestra()`: filtros sobre el texto (cuentas de traducción, encabezados y comillas,
   longitud mínima, duplicados, idioma) y selección de `n_por_genero` canciones por género.

Uso:  python -m src.muestra
Alrededor de 25 minutos en la primera ejecución; los pasos cuyo archivo ya existe se omiten.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.datos import GENEROS, RAIZ, SEMILLA, leer_por_partes, limpiar_letra

SALIDA = RAIZ / "data" / "processed"
_ENCABEZADO = r"\[[^\]\n]*\]"

# Filas del CSV (identificadores de 754 mil a 2,31 millones, aproximadamente) provenientes de una
# importación masiva de catálogo en la que "pop" funciona como etiqueta por defecto: Status Quo,
# B.B. King o Los Tigres del Norte figuran como pop. El bloque contiene el 43% de todo el pop.
# Los límites corresponden a los puntos en que la proporción móvil de pop cruza el 70%
# (notebooks/01_eda.ipynb, sección 1.2).
BLOQUE_IMPORTADO = (470_000, 1_745_000)
# Por debajo de este umbral predominan marcadores ("Coming soon", "[Instrumental]") y fragmentos.
MIN_PALABRAS = 20
# Cuentas de Genius que publican traducciones y romanizaciones ("Genius English Translations",
# "Genius Romanizations", etc.). Sus textos no son letras originales y el valor de `artist`
# agrupa canciones de muchos artistas distintos.
CUENTAS_GENIUS = r"Genius\b"


def _hash_letra(letra: str) -> int:
    """Hash de 64 bits de la letra en minúsculas, para detectar duplicados exactos.

    No normaliza los espacios internos, porque hacerlo sobre 9 GB de texto multiplica el tiempo
    de cómputo; la deduplicación que ignora puntuación y espacios se aplica sobre la muestra.
    """
    clave = letra.strip().lower().encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.blake2b(clave, digest_size=8).digest(), "little")


def _no_ascii(letra: str) -> int:
    return len(letra) - len(letra.encode("ascii", "ignore"))


def _metadatos(parte: pd.DataFrame, fila_inicial: int) -> pd.DataFrame:
    letras = parte["lyrics"].fillna("")
    return pd.DataFrame(
        {
            "fila": np.arange(fila_inicial, fila_inicial + len(parte)),  # posición en el CSV
            "id": parte["id"].to_numpy(),
            "tag": parte["tag"].astype("string"),
            "artist": parte["artist"].astype("string"),
            "year": parte["year"].astype("Int32"),
            "views": parte["views"].astype("Int64"),
            "con_features": parte["features"].fillna("{}").str.strip().ne("{}"),
            "n_chars": letras.str.len().astype("int32"),
            "n_no_ascii": letras.map(_no_ascii).astype("int32"),
            "n_encabezados": letras.str.count(_ENCABEZADO).astype("int32"),
            "hash_letra": letras.map(_hash_letra).astype("uint64"),
        }
    )


def construir_metadatos(**kwargs) -> None:
    """Paso 1: genera metadatos.parquet, con una fila por canción del corpus completo."""
    SALIDA.mkdir(parents=True, exist_ok=True)
    escritor = None
    fila = 0
    t0 = time.time()
    for i, parte in enumerate(leer_por_partes(**kwargs)):
        tabla = pa.Table.from_pandas(_metadatos(parte, fila), preserve_index=False)
        if escritor is None:
            escritor = pq.ParquetWriter(SALIDA / "metadatos.parquet", tabla.schema)
        escritor.write_table(tabla)
        fila += len(parte)
        print(f"parte {i:2d}  filas {fila:>9,}  {time.time() - t0:5.0f}s", flush=True)
    escritor.close()


def elegir_candidatas(meta: pd.DataFrame, n: int = 22_000, semilla: int = SEMILLA) -> pd.DataFrame:
    """Paso 2a: filtra los metadatos y sortea `n` canciones por género, sin reemplazo.

    Descarta los géneros fuera de GENEROS, el bloque de importación y toda canción cuyo texto
    exacto aparece más de una vez en el corpus (marcadores como "[Instrumental]" y copias entre
    artistas o géneros).
    """
    repetidas = meta["hash_letra"].duplicated(keep=False)
    ok = (
        meta["tag"].isin(GENEROS)
        & ~meta["fila"].between(*BLOQUE_IMPORTADO)
        & ~repetidas
    )
    return (
        meta[ok]
        .sample(frac=1, random_state=semilla)
        .groupby("tag")
        .head(n)
        .sort_values("fila")
        .reset_index(drop=True)
    )


def extraer_letras(filas, **kwargs) -> pd.DataFrame:
    """Paso 2b: recorre el archivo comprimido y devuelve, con su letra, solo las filas indicadas."""
    buscadas = np.sort(np.asarray(filas))
    partes, inicio = [], 0
    for parte in leer_por_partes(**kwargs):
        idx = np.arange(inicio, inicio + len(parte))
        mascara = np.isin(idx, buscadas)
        partes.append(parte[mascara].assign(fila=idx[mascara]))
        inicio += len(parte)
    return pd.concat(partes, ignore_index=True)


def detectar_idioma(textos: list[str], largo: int = 1000) -> list[str]:
    """Código ISO 639-1 del idioma de cada texto, estimado con lingua en modo de alta precisión.

    Se analizan los primeros `largo` caracteres de cada texto.
    """
    from lingua import LanguageDetectorBuilder

    detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    idiomas = detector.detect_languages_in_parallel_of([t[:largo] for t in textos])
    return [i.iso_code_639_1.name.lower() if i is not None else "?" for i in idiomas]


def preparar_muestra(
    candidatas: pd.DataFrame, n_por_genero: int = 10_000, semilla: int = SEMILLA
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paso 3: aplica los filtros de texto y selecciona `n_por_genero` canciones por género.

    `candidatas` debe incluir la columna `idioma` (ver `detectar_idioma`). Devuelve la muestra y
    una tabla con la cantidad de canciones por género que quedan después de cada filtro.
    """
    df = candidatas.sort_values("fila").reset_index(drop=True)
    # Cada canción recibe una clave aleatoria antes de filtrar y la selección final toma las claves
    # más bajas de cada género. Así, modificar un filtro solo altera las canciones afectadas por él.
    df["azar"] = np.random.default_rng(semilla).random(len(df))
    df["letra"] = df["lyrics"].fillna("").map(limpiar_letra)
    df["n_palabras"] = df["letra"].str.split().str.len()
    # Clave de duplicado que ignora mayúsculas, puntuación y espacios.
    df["clave"] = df["letra"].str.lower().str.replace(r"[\W_]+", "", regex=True)

    pasos = {"candidatas": df}
    df = df[~df["artist"].fillna("").str.match(CUENTAS_GENIUS)]
    pasos["sin cuentas de traducción de Genius"] = df
    df = df[df["n_palabras"] >= MIN_PALABRAS]
    pasos[f">= {MIN_PALABRAS} palabras sin encabezados"] = df
    df = df[~df["clave"].duplicated(keep=False)]
    pasos["sin duplicados (puntuación/espacios)"] = df
    df = df[df["idioma"] == "en"]
    pasos["en inglés"] = df
    df = (
        df.sort_values("azar")
        .groupby("tag")
        .head(n_por_genero)
        .sort_values(["tag", "fila"])
        .reset_index(drop=True)
    )
    pasos[f"recorte a {n_por_genero:,} por género"] = df

    embudo = pd.DataFrame({k: v["tag"].value_counts() for k, v in pasos.items()}).T[GENEROS]
    return df.drop(columns=["clave", "azar"]), embudo


if __name__ == "__main__":
    if not (SALIDA / "metadatos.parquet").exists():
        construir_metadatos()
    if not (SALIDA / "candidatas.parquet").exists():
        meta = pd.read_parquet(SALIDA / "metadatos.parquet")
        elegidas = elegir_candidatas(meta)
        # Primera aparición de los 30 textos más repetidos del corpus (notebook, sección 1.3).
        repetidos = meta["hash_letra"].value_counts().head(30)
        filas_rep = meta.drop_duplicates("hash_letra").set_index("hash_letra").loc[repetidos.index, "fila"]
        del meta
        letras = extraer_letras(np.concatenate([elegidas["fila"], filas_rep]))
        marcadores = letras[letras["fila"].isin(filas_rep)]
        marcadores = marcadores[["fila", "lyrics"]].assign(
            veces=marcadores["fila"].map(dict(zip(filas_rep, repetidos)))
        )
        marcadores.sort_values("veces", ascending=False).to_parquet(SALIDA / "marcadores.parquet", index=False)
        candidatas = letras[letras["fila"].isin(elegidas["fila"])].reset_index(drop=True)
        candidatas["idioma"] = detectar_idioma(candidatas["lyrics"].fillna("").map(limpiar_letra).tolist())
        candidatas.to_parquet(SALIDA / "candidatas.parquet", index=False)
    candidatas = pd.read_parquet(SALIDA / "candidatas.parquet")
    muestra, embudo = preparar_muestra(candidatas)
    muestra.to_parquet(SALIDA / "muestra.parquet", index=False)
    print(embudo.to_string())
