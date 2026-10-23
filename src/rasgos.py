"""Rasgos de forma y estilo de cada letra.

Unos 40 rasgos interpretables, organizados en grupos que son la unidad de la ablación (GRUPOS):

    E  estructura           R  repetición          S  rima y sonido       M  métrica
    D  diversidad léxica    P  pronombres          O  oralidad y registro X  sintaxis
    T  transcripción (control: convenciones de quien transcribe en Genius, no lenguaje)

E, R, S, M y D describen la forma sin mirar qué palabras se usan; P, O y X describen el estilo a partir de
listas cerradas de palabras o de categorías gramaticales, por lo que están más cerca del contenido.

Vocabulario: "verso" es una línea de la letra y "estrofa", un bloque de versos separado por una línea vacía.
Las pronunciaciones salen del diccionario CMU (nltk); las palabras que no figuran en él (jerga, nombres)
no participan de las medidas de rima, y su proporción se informa como `o_cobertura_cmu`.

Uso:  python -m src.rasgos     genera data/processed/rasgos.parquet (alrededor de 10 minutos)
"""
from __future__ import annotations

import re
import time
import zlib
from collections import Counter
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.datos import RAIZ

SALIDA = RAIZ / "data" / "processed" / "rasgos.parquet"

GRUPOS = {
    "E": ["e_log_palabras", "e_log_versos", "e_palabras_por_verso", "e_cv_palabras_por_verso", "e_estrofas",
          "e_versos_por_estrofa"],
    "R": ["r_pct_versos_repetidos", "r_compresion", "r_estribillo", "r_trigramas_repetidos"],
    "S": ["s_rima_AA", "s_rima_ABAB", "s_densidad_rima", "s_densidad_multisilabica", "s_densidad_asonancia",
          "s_rima_interna", "s_aliteracion"],
    "M": ["m_silabas_por_verso", "m_cv_silabas", "m_pct_acentuadas"],
    "D": ["d_tipos_100", "d_mtld", "d_hapax", "d_long_palabra"],
    "P": ["p_yo", "p_tu", "p_nosotros", "p_tercera", "p_log_yo_tu"],
    "O": ["o_contracciones", "o_apocopes", "o_vocalizaciones", "o_coloquial", "o_groserias", "o_cobertura_cmu"],
    "X": ["x_sustantivos", "x_verbos", "x_adjetivos", "x_adverbios", "x_pasado", "x_negacion"],
    "T": ["t_pct_mayusculas", "t_pct_parentesis", "t_comas_por_verso", "t_signos_por_verso"],
}
NOMBRE_GRUPO = {"E": "estructura", "R": "repetición", "S": "rima y sonido", "M": "métrica",
                "D": "diversidad léxica", "P": "pronombres", "O": "oralidad y registro", "X": "sintaxis",
                "T": "transcripción (control)"}
LINGUISTICOS = [c for g, cols in GRUPOS.items() if g != "T" for c in cols]

TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*'?")       # incluye el apóstrofo final de las apócopes: nothin'
VENTANA_RIMA = 4                                  # versos anteriores con los que puede rimar un verso
FUNCIONALES = ENGLISH_STOP_WORDS

YO = {"i", "me", "my", "mine", "myself", "i'm", "i'll", "i've", "i'd"}
TU = {"you", "your", "yours", "yourself", "you're", "you'll", "you've", "you'd", "ya", "u"}
NOSOTROS = {"we", "us", "our", "ours", "ourselves", "we're", "we'll", "we've", "we'd"}
TERCERA = {"he", "she", "him", "her", "his", "hers", "they", "them", "their", "theirs", "he's", "she's",
           "they're", "he'll", "she'll", "they'll"}
COLOQUIAL = {"ain't", "gonna", "wanna", "gotta", "tryna", "finna", "imma", "y'all", "yall", "lemme", "gimme",
             "kinda", "outta", "lotta", "cuz", "cause", "nah", "em", "gon", "bout", "til", "tho", "dat", "dem"}
GROSERIAS = {"fuck", "fucking", "shit", "bitch", "bitches", "damn", "ass", "nigga", "niggas"}   # lista del EDA
NEGACION = {"not", "no", "never", "nothing", "nobody", "nowhere", "none", "nor", "neither", "nothin'"}
VOCALIZACION = re.compile(
    r"^(?:o+h*|a+h+|u+h+|e+h+|ye+a*h*|hey+|la+|na+|da+|wo+a*h*|who+a+h*|ay+e*|mm+|hm+|oo+h*|ha+)$"
)

_CMU: dict | None = None


# ----------------------------------------------------------------------------- pronunciación
def _cmu() -> dict:
    global _CMU
    if _CMU is None:
        import nltk
        try:
            from nltk.corpus import cmudict
            _CMU = cmudict.dict()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            from nltk.corpus import cmudict
            _CMU = cmudict.dict()
    return _CMU


@lru_cache(maxsize=None)
def pronunciacion(palabra: str) -> tuple[str, ...] | None:
    """Fonemas ARPAbet de la primera pronunciación del CMU, o None si la palabra no figura.

    Prueba también la forma completa de las apócopes ("nothin'" -> "nothing") y la palabra sin apóstrofos.
    """
    cmu = _cmu()
    w = palabra.strip("'")
    for cand in (w, w + "g" if w.endswith("in") else None, w.replace("'", "")):
        if cand and cand in cmu:
            return tuple(cmu[cand][0])
    return None


def _es_vocal(fonema: str) -> bool:
    return fonema[-1].isdigit()


def _sin_acento(fonema: str) -> str:
    return fonema.rstrip("012")


def clave_rima(fonemas: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Rima perfecta: fonemas desde la última vocal acentuada hasta el final, sin marcas de acento."""
    if not fonemas:
        return None
    vocales = [i for i, f in enumerate(fonemas) if _es_vocal(f)]
    if not vocales:
        return None
    acentuadas = [i for i in vocales if fonemas[i][-1] in "12"]
    return tuple(_sin_acento(f) for f in fonemas[(acentuadas or vocales)[-1]:])


def _final_de_verso(palabras: list[str]) -> tuple[str, ...] | None:
    """Fonemas de las dos últimas palabras del verso (para rimas que abarcan más de una palabra)."""
    fon = [pronunciacion(w) for w in palabras[-2:]]
    if fon[-1] is None:
        return None
    return sum((f for f in fon if f is not None), ())


def clave_multisilabica(fonemas: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Fonemas desde la penúltima vocal: "basement" / "placement"; puede abarcar dos palabras."""
    if not fonemas:
        return None
    vocales = [i for i, f in enumerate(fonemas) if _es_vocal(f)]
    return tuple(_sin_acento(f) for f in fonemas[vocales[-2]:]) if len(vocales) >= 2 else None


def clave_asonancia(fonemas: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Solo las dos últimas vocales: rima imperfecta típica del rap, "basement" / "patience"."""
    if not fonemas:
        return None
    vocales = [_sin_acento(f) for f in fonemas if _es_vocal(f)]
    return tuple(vocales[-2:]) if len(vocales) >= 2 else None


def silabas(palabra: str) -> int:
    """Sílabas según el CMU; para palabras fuera del diccionario, grupos de vocales escritas."""
    fon = pronunciacion(palabra)
    if fon:
        return sum(_es_vocal(f) for f in fon)
    return max(1, len(re.findall(r"[aeiouy]+", palabra)))


# ----------------------------------------------------------------------------- medidas auxiliares
def _media(x) -> float:
    return float(np.mean(x)) if len(x) else np.nan


def _cv(x) -> float:
    m = np.mean(x) if len(x) else 0
    return float(np.std(x) / m) if m > 0 else np.nan


def _tasa_pares(claves: list, finales: list, salto: int) -> float:
    """Proporción de pares de versos a distancia `salto` cuyas palabras finales riman.

    Se descartan los pares que terminan con la misma palabra: repetirla no es rimar.
    """
    pares = [a == b for a, b, wa, wb in zip(claves, claves[salto:], finales, finales[salto:])
             if a and b and wa != wb]
    return _media(pares) if len(pares) >= 3 else np.nan


def _densidad(claves: list, finales: list, ventana: int = VENTANA_RIMA) -> float:
    """Proporción de versos cuyo final rima con el de alguno de los `ventana` versos anteriores."""
    aciertos = []
    for i in range(1, len(claves)):
        if not claves[i]:
            continue
        previas = [claves[j] for j in range(max(0, i - ventana), i) if claves[j] and finales[j] != finales[i]]
        if previas:
            aciertos.append(claves[i] in previas)
    return _media(aciertos) if len(aciertos) >= 3 else np.nan


def _rima_interna(versos: list[list[str]]) -> float:
    """Proporción de versos con al menos dos palabras de contenido distintas que riman entre sí."""
    aciertos = []
    for palabras in versos:
        contenido = [w for w in palabras if len(w) >= 3 and w not in FUNCIONALES]
        if len(contenido) < 3:
            continue
        vistas, rima = {}, False
        for w in contenido:
            k = clave_rima(pronunciacion(w))
            if k is None:
                continue
            if k in vistas and vistas[k] != w:
                rima = True
                break
            vistas.setdefault(k, w)
        aciertos.append(rima)
    return _media(aciertos) if len(aciertos) >= 3 else np.nan


def _aliteracion(versos: list[list[str]]) -> float:
    """Proporción de pares de palabras de contenido consecutivas que empiezan con la misma consonante."""
    pares = []
    for palabras in versos:
        inicios = []
        for w in palabras:
            if len(w) < 3 or w in FUNCIONALES:
                continue
            fon = pronunciacion(w)
            inicios.append(fon[0] if fon and not _es_vocal(fon[0]) else None)
        pares += [a == b for a, b in zip(inicios, inicios[1:]) if a and b]
    return _media(pares) if len(pares) >= 5 else np.nan


def _cobertura_estribillo(versos_norm: list[str], max_largo: int = 8) -> float:
    """Fracción de los versos cubierta por el bloque de 2 o más versos consecutivos que más se repite.

    Para cada largo se toma el bloque más frecuente y se cuentan sus apariciones sin superposición.
    """
    n, mejor = len(versos_norm), 0.0
    for largo in range(2, min(max_largo, n // 2) + 1):
        bloques = [tuple(versos_norm[i:i + largo]) for i in range(n - largo + 1)]
        bloque, _ = Counter(bloques).most_common(1)[0]
        apariciones, i = 0, 0
        while i <= n - largo:
            if bloques[i] == bloque:
                apariciones += 1
                i += largo
            else:
                i += 1
        if apariciones >= 2:
            mejor = max(mejor, largo * apariciones / n)
    return mejor


def mtld(tokens: list[str], umbral: float = 0.72) -> float:
    """Medida de diversidad léxica MTLD (McCarthy y Jarvis, 2010), promedio de ambas direcciones.

    A diferencia de la razón tipos/tokens, no depende de la longitud del texto.
    """
    if len(tokens) < 50:
        return np.nan

    def una_direccion(toks):
        factores, tipos, n = 0.0, set(), 0
        for w in toks:
            tipos.add(w)
            n += 1
            if len(tipos) / n <= umbral:
                factores, tipos, n = factores + 1, set(), 0
        if n:
            factores += (1 - len(tipos) / n) / (1 - umbral)
        return len(toks) / factores if factores else np.nan

    return float(np.nanmean([una_direccion(tokens), una_direccion(tokens[::-1])]))


# ----------------------------------------------------------------------------- rasgos de una letra
def rasgos_letra(letra: str) -> dict[str, float]:
    """Todos los rasgos de una letra salvo los de categorías gramaticales (ver `rasgos_sintaxis`)."""
    lineas = letra.split("\n")
    versos_txt = [l.strip() for l in lineas if l.strip()]
    versos = [TOKEN.findall(v.lower()) for v in versos_txt]
    versos = [v for v in versos if v]
    toks = [w for v in versos for w in v]
    n, nv = len(toks), len(versos)
    if n == 0:
        return {c: np.nan for c in LINGUISTICOS + GRUPOS["T"] if not c.startswith("x_") or c == "x_negacion"}

    # Estructura
    estrofas = [len(b) for b in re.split(r"\n\s*\n", letra.strip()) if b.strip()]
    por_verso = [len(v) for v in versos]
    r = {
        "e_log_palabras": np.log1p(n),
        "e_log_versos": np.log1p(nv),
        "e_palabras_por_verso": n / nv,
        "e_cv_palabras_por_verso": _cv(por_verso),
        "e_estrofas": len(estrofas),
        "e_versos_por_estrofa": nv / max(len(estrofas), 1),
    }

    # Repetición
    norm = [" ".join(v) for v in versos]
    trigramas = list(zip(toks, toks[1:], toks[2:]))
    texto = " ".join(norm).encode("utf-8")
    r |= {
        "r_pct_versos_repetidos": 100 * (1 - len(set(norm)) / nv),
        "r_compresion": len(zlib.compress(texto, 9)) / len(texto),
        "r_estribillo": _cobertura_estribillo(norm),
        "r_trigramas_repetidos": 1 - len(set(trigramas)) / len(trigramas) if len(trigramas) >= 10 else np.nan,
    }

    # Rima y sonido
    finales = [v[-1] for v in versos]
    fon_final = [pronunciacion(w) for w in finales]
    claves = [clave_rima(f) for f in fon_final]
    fon_dos = [_final_de_verso(v) for v in versos]
    r |= {
        "s_rima_AA": _tasa_pares(claves, finales, 1),
        "s_rima_ABAB": _tasa_pares(claves, finales, 2),
        "s_densidad_rima": _densidad(claves, finales),
        "s_densidad_multisilabica": _densidad([clave_multisilabica(f) for f in fon_dos], finales),
        "s_densidad_asonancia": _densidad([clave_asonancia(f) for f in fon_dos], finales),
        "s_rima_interna": _rima_interna(versos),
        "s_aliteracion": _aliteracion(versos),
    }

    # Métrica
    sil = [sum(silabas(w) for w in v) for v in versos]
    vocales = [f for w in toks if (fon := pronunciacion(w)) for f in fon if _es_vocal(f)]
    r |= {
        "m_silabas_por_verso": _media(sil),
        "m_cv_silabas": _cv(sil),
        "m_pct_acentuadas": _media([f[-1] in "12" for f in vocales]),
    }

    # Diversidad léxica
    cuenta = Counter(toks)
    r |= {
        "d_tipos_100": len(set(toks[:100])) if n >= 100 else np.nan,
        "d_mtld": mtld(toks),
        "d_hapax": sum(c == 1 for c in cuenta.values()) / len(cuenta),
        "d_long_palabra": _media([len(w.strip("'")) for w in toks]),
    }

    # Pronombres (por cada 100 palabras)
    def tasa(conjunto):
        return 100 * sum(cuenta[w] for w in conjunto) / n

    n_yo, n_tu = sum(cuenta[w] for w in YO), sum(cuenta[w] for w in TU)
    r |= {
        "p_yo": tasa(YO), "p_tu": tasa(TU), "p_nosotros": tasa(NOSOTROS), "p_tercera": tasa(TERCERA),
        "p_log_yo_tu": np.log((n_yo + 0.5) / (n_tu + 0.5)),
    }

    # Oralidad y registro (por cada 100 palabras, salvo la cobertura)
    apocope = [w.endswith("in'") or (w.endswith("in") and w not in _cmu() and w + "g" in _cmu()) for w in toks]
    r |= {
        "o_contracciones": 100 * _media(["'" in w.strip("'") for w in toks]),
        "o_apocopes": 100 * _media(apocope),
        "o_vocalizaciones": 100 * _media([bool(VOCALIZACION.match(w)) for w in toks]),
        "o_coloquial": tasa(COLOQUIAL),
        "o_groserias": tasa(GROSERIAS),
        "o_cobertura_cmu": _media([pronunciacion(w) is not None for w in toks]),
        "x_negacion": 100 * _media([w in NEGACION or w.endswith("n't") for w in toks]),
    }

    # Transcripción (control): dependen de quien transcribe, no de la canción
    palabras_orig = [w for w in letra.split() if re.search(r"[A-Za-z]", w)]
    r |= {
        "t_pct_mayusculas": 100 * _media([w.isupper() and len(w) > 1 for w in palabras_orig]),
        "t_pct_parentesis": 100 * _media(["(" in v for v in versos_txt]),
        "t_comas_por_verso": sum(v.count(",") for v in versos_txt) / nv,
        "t_signos_por_verso": sum(len(re.findall(r"[.!?;:]", v)) for v in versos_txt) / nv,
    }
    return r


# ----------------------------------------------------------------------------- sintaxis (spaCy)
def rasgos_sintaxis(letras: Iterable[str], n_process: int = 1, batch_size: int = 64) -> pd.DataFrame:
    """Proporción de sustantivos, verbos, adjetivos y adverbios, y de verbos en pasado, con spaCy.

    Requiere el modelo en_core_web_sm (`python -m spacy download en_core_web_sm`).
    """
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    filas = []
    for doc in nlp.pipe(letras, n_process=n_process, batch_size=batch_size):
        toks = [t for t in doc if not (t.is_space or t.is_punct)]
        n = max(len(toks), 1)
        pos = Counter(t.pos_ for t in toks)
        verbos = [t for t in toks if t.tag_.startswith("VB")]
        filas.append({
            "x_sustantivos": (pos["NOUN"] + pos["PROPN"]) / n,
            "x_verbos": pos["VERB"] / n,
            "x_adjetivos": pos["ADJ"] / n,
            "x_adverbios": pos["ADV"] / n,
            "x_pasado": sum(t.tag_ in ("VBD", "VBN") for t in verbos) / len(verbos) if verbos else np.nan,
        })
    return pd.DataFrame(filas)


def extraer(muestra: pd.DataFrame, sintaxis: bool = True, n_process: int = 1) -> pd.DataFrame:
    """Tabla de rasgos con una fila por canción (columna `id`) y las columnas de GRUPOS."""
    t0 = time.time()
    tabla = pd.DataFrame([rasgos_letra(t) for t in muestra["letra"]])
    print(f"rasgos de forma: {time.time() - t0:.0f} s", flush=True)
    if sintaxis:
        t0 = time.time()
        tabla = pd.concat([tabla, rasgos_sintaxis(muestra["letra"], n_process=n_process)], axis=1)
        print(f"sintaxis (spaCy): {time.time() - t0:.0f} s", flush=True)
    tabla.insert(0, "id", muestra["id"].to_numpy())
    columnas = ["id"] + [c for cols in GRUPOS.values() for c in cols if c in tabla]
    return tabla[columnas]


if __name__ == "__main__":
    import os

    muestra = pd.read_parquet(RAIZ / "data" / "processed" / "muestra.parquet", columns=["id", "letra"])
    tabla = extraer(muestra, n_process=max(1, (os.cpu_count() or 2) - 1))
    tabla.to_parquet(SALIDA, index=False)
    print(f"{tabla.shape[0]:,} canciones × {tabla.shape[1] - 1} rasgos -> {SALIDA}")
    print("faltantes por rasgo (%):")
    print((tabla.isna().mean() * 100).round(1).loc[lambda s: s > 0].to_string())
