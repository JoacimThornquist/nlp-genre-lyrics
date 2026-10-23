"""Protocolo común de evaluación de todos los modelos.

- Particiones: validación cruzada estratificada y agrupada por artista, calculada una sola vez y guardada en
  data/processed/folds.parquet. Todos los modelos, locales o en Colab, la leen de ese archivo, por lo que las
  comparaciones entre modelos son apareadas.
- Predicciones fuera de muestra (OOF): cada modelo guarda en data/processed/oof/<modelo>.parquet la
  probabilidad de cada género para cada canción, calculada por un modelo que no vio ni la canción ni a su
  artista. Las comparaciones del informe se calculan a partir de esos archivos.
- Métricas: F1 por género y macro-F1, con intervalos de confianza por bootstrap de artistas.
- Registro: cada corrida agrega una fila a reports/tablas/registro_experimentos.csv.

Uso:  python -m src.evaluacion
Crea folds.parquet si no existe y corre las referencias M0 (clase mayoritaria) y M1 (TF-IDF + regresión
logística). Tarda alrededor de 10 minutos.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src.datos import GENEROS, RAIZ, SEMILLA

PROC = RAIZ / "data" / "processed"
RUTA_MUESTRA = PROC / "muestra.parquet"
RUTA_FOLDS = PROC / "folds.parquet"
DIR_OOF = PROC / "oof"
REGISTRO = RAIZ / "reports" / "tablas" / "registro_experimentos.csv"

N_FOLDS = 5
K = len(GENEROS)
COLS_PROBA = [f"p_{g}" for g in GENEROS]
# Configuración del clasificador de referencia M1, idéntica a la del notebook 01.
TFIDF_REFERENCIA = dict(min_df=3, max_features=100_000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")


# ----------------------------------------------------------------------------- particiones
def etiquetas(tags: Iterable[str]) -> np.ndarray:
    """Etiquetas como enteros 0..K-1, en el orden de GENEROS."""
    codigos = pd.Categorical(np.asarray(tags), categories=GENEROS).codes
    if (codigos < 0).any():
        raise ValueError("hay etiquetas fuera de GENEROS")
    return codigos.astype(int)


def crear_folds(muestra: pd.DataFrame, n_folds: int = N_FOLDS, semilla: int = SEMILLA) -> pd.DataFrame:
    """Asigna cada canción a una de `n_folds` particiones estratificadas por género y agrupadas por artista.

    Los grupos se pasan como texto, igual que en el notebook 01, para que las particiones coincidan con las
    de la validación cruzada del clasificador de referencia (macro-F1 0,589 ± 0,005).
    """
    divisor = StratifiedGroupKFold(n_folds, shuffle=True, random_state=semilla)
    fold = np.empty(len(muestra), dtype="int8")
    for k, (_, i_te) in enumerate(divisor.split(muestra, etiquetas(muestra["tag"]), muestra["artist"].to_numpy())):
        fold[i_te] = k
    return pd.DataFrame({"id": muestra["id"].to_numpy(), "artist": muestra["artist"].to_numpy(), "fold": fold})


def guardar_folds(folds: pd.DataFrame) -> None:
    compartidos = folds.groupby("artist")["fold"].nunique().gt(1).sum()
    assert compartidos == 0, f"{compartidos} artistas en más de una partición"
    RUTA_FOLDS.parent.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(RUTA_FOLDS, index=False)


def cargar_folds(ids: Iterable) -> np.ndarray:
    """Partición de cada canción de `ids`, en ese mismo orden."""
    fold = pd.read_parquet(RUTA_FOLDS).set_index("id")["fold"]
    ids = pd.Index(ids)
    faltan = ~ids.isin(fold.index)
    if faltan.any():
        raise ValueError(f"{faltan.sum()} canciones sin partición; si cambió la muestra, regenerar folds.parquet")
    return fold.loc[ids].to_numpy()


def particiones(fold: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pares (índices de entrenamiento, índices de evaluación), uno por partición."""
    return [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(int(fold.max()) + 1)]


def division_interna(i_tr: np.ndarray, grupos: Iterable, frac_val: float = 0.15, semilla: int = SEMILLA):
    """Separa del entrenamiento una validación con artistas propios, para early stopping o elegir épocas.

    La partición de evaluación nunca se usa para decidir nada del entrenamiento.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=frac_val, random_state=semilla)
    a, b = next(gss.split(i_tr, groups=np.asarray(grupos)[i_tr]))
    return i_tr[a], i_tr[b]


# ----------------------------------------------------------------------------- predicciones fuera de muestra
def _sel(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def con_sklearn(fabrica: Callable, X, y_texto: np.ndarray) -> Callable:
    """Adapta un estimador de scikit-learn a la interfaz de `cv_oof`."""
    y_texto = np.asarray(y_texto)

    def entrenar_predecir(i_tr, i_te):
        m = fabrica().fit(_sel(X, i_tr), y_texto[i_tr])
        proba = np.zeros((len(i_te), K))
        proba[:, [GENEROS.index(c) for c in m.classes_]] = m.predict_proba(_sel(X, i_te))
        return proba

    return entrenar_predecir


def predecir(proba: np.ndarray) -> np.ndarray:
    """Género más probable de cada fila; -1 en las filas sin predicción."""
    pred = np.full(len(proba), -1)
    ok = ~np.isnan(proba).all(1)
    pred[ok] = np.nan_to_num(proba[ok], nan=-1).argmax(1)
    return pred


def cv_oof(entrenar_predecir: Callable, Y: np.ndarray, fold: np.ndarray, solo_folds: Iterable[int] | None = None,
           nombre: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades fuera de muestra (n, K) y macro-F1 de cada partición evaluada.

    `entrenar_predecir(i_tr, i_te)` entrena con las canciones `i_tr` y devuelve las probabilidades de las
    canciones `i_te` en el orden de GENEROS. `solo_folds` limita las particiones (p. ej. [0] para modelos
    costosos); las canciones de las particiones no evaluadas quedan con NaN.
    """
    proba = np.full((len(Y), K), np.nan)
    f1s, t0 = [], time.time()
    for k, (i_tr, i_te) in enumerate(particiones(fold)):
        if solo_folds is not None and k not in solo_folds:
            continue
        proba[i_te] = entrenar_predecir(i_tr, i_te)
        f1s.append(macro_f1(Y[i_te], predecir(proba[i_te])))
        print(f"  {nombre} partición {k}: macro-F1 {f1s[-1]:.3f}  ({time.time() - t0:.0f} s)", flush=True)
    f1s = np.array(f1s)
    print(f"{nombre}: macro-F1 {f1s.mean():.3f} ± {f1s.std():.3f} en {len(f1s)} particiones", flush=True)
    return proba, f1s


def guardar_oof(nombre: str, ids: Iterable, fold: np.ndarray, proba: np.ndarray) -> None:
    """Guarda las filas con predicción en data/processed/oof/<nombre>.parquet."""
    ok = ~np.isnan(proba).all(1)
    df = pd.DataFrame(proba[ok], columns=COLS_PROBA)
    df.insert(0, "fold", np.asarray(fold)[ok])
    df.insert(0, "id", np.asarray(ids)[ok])
    df["pred"] = np.array(GENEROS)[predecir(proba[ok])]
    DIR_OOF.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DIR_OOF / f"{nombre}.parquet", index=False)


def cargar_oof(nombre: str, ids: Iterable) -> np.ndarray:
    """Probabilidades (n, K) de un modelo, alineadas con `ids`; NaN donde no hay predicción."""
    df = pd.read_parquet(DIR_OOF / f"{nombre}.parquet").set_index("id")
    return df.reindex(pd.Index(ids))[COLS_PROBA].to_numpy()


def modelos_guardados() -> list[str]:
    return sorted(p.stem for p in DIR_OOF.glob("*.parquet"))


# ----------------------------------------------------------------------------- métricas
def f1_clases(Y: np.ndarray, P: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """F1 de cada género; `w` son pesos por canción (bootstrap por artista)."""
    cm = np.bincount(Y * K + P, weights=w, minlength=K * K).reshape(K, K)
    tp, n_pred, n_real = np.diag(cm), cm.sum(0), cm.sum(1)
    return np.divide(2 * tp, n_pred + n_real, out=np.zeros(K), where=(n_pred + n_real) > 0)


def macro_f1(Y: np.ndarray, P: np.ndarray, w: np.ndarray | None = None) -> float:
    """Promedio del F1 de los géneros presentes en `Y`."""
    return float(f1_clases(Y, P, w)[np.unique(Y)].mean())


def bootstrap_artistas(preds: dict[str, np.ndarray], Y: np.ndarray, grupos: Iterable, n_boot: int = 1000,
                       semilla: int = SEMILLA) -> dict[str, np.ndarray]:
    """F1 por género y macro-F1 de cada modelo en `n_boot` remuestreos de artistas.

    Se remuestrean artistas, no canciones, porque las canciones de un mismo artista no son independientes:
    cada artista recibe un peso entero (cuántas veces salió sorteado) y sus canciones lo heredan. Los
    remuestreos son los mismos para todos los modelos, así las diferencias entre modelos son apareadas.
    Devuelve, para cada modelo, un arreglo (n_boot, K + 1) cuya última columna es el macro-F1.
    """
    codigos, unicos = pd.factorize(pd.Series(np.asarray(grupos)))
    rng = np.random.default_rng(semilla)
    clases = np.unique(Y)
    salida = {n: np.zeros((n_boot, K + 1)) for n in preds}
    for b in range(n_boot):
        w = rng.multinomial(len(unicos), np.full(len(unicos), 1 / len(unicos)))[codigos].astype(float)
        for n, P in preds.items():
            f = f1_clases(Y, P, w)
            salida[n][b, :K], salida[n][b, K] = f, f[clases].mean()
    return salida


def ic(x: np.ndarray, nivel: float = 95) -> np.ndarray:
    """Intervalo percentil por columna: arreglo (2, columnas)."""
    return np.percentile(x, [(100 - nivel) / 2, 100 - (100 - nivel) / 2], axis=0)


def comparar(probas: dict[str, np.ndarray], Y: np.ndarray, grupos: Iterable, referencia: str | None = None,
             n_boot: int = 1000) -> pd.DataFrame:
    """Tabla de comparación sobre las canciones que todos los modelos predijeron.

    Para cada modelo: macro-F1 con IC 95%, F1 por género y, si se indica `referencia`, la diferencia apareada
    de macro-F1 respecto de ese modelo con su IC 95%.
    """
    comunes = np.logical_and.reduce([~np.isnan(p).all(1) for p in probas.values()])
    preds = {n: predecir(p[comunes]) for n, p in probas.items()}
    Yc, gc = Y[comunes], np.asarray(grupos)[comunes]
    boot = bootstrap_artistas(preds, Yc, gc, n_boot=n_boot)
    filas = []
    for n, P in preds.items():
        f = f1_clases(Yc, P)
        lo, hi = ic(boot[n][:, K])
        fila = {"modelo": n, "canciones": int(comunes.sum()), "macro-F1": f[np.unique(Yc)].mean(),
                "IC inf": lo, "IC sup": hi, **dict(zip(GENEROS, f))}
        if referencia is not None and n != referencia:
            d = boot[n][:, K] - boot[referencia][:, K]
            fila["Δ vs " + referencia], (fila["Δ IC inf"], fila["Δ IC sup"]) = d.mean(), ic(d)
        filas.append(fila)
    return pd.DataFrame(filas).set_index("modelo")


# ----------------------------------------------------------------------------- registro
def _commit() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, capture_output=True, text=True,
                           check=True).stdout.strip()
        sucio = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src"], cwd=RAIZ).returncode != 0
        return h + ("+cambios" if sucio else "")
    except (OSError, subprocess.CalledProcessError):
        return "?"


def registrar(modelo: str, entrada: str, hiperparametros: dict, f1_folds: np.ndarray, segundos: float,
              notas: str = "") -> None:
    """Agrega una fila a reports/tablas/registro_experimentos.csv.

    El commit lleva el sufijo "+cambios" si había modificaciones sin commitear en src/.
    """
    fila = pd.DataFrame([{
        "fecha": datetime.now().isoformat(timespec="minutes"),
        "modelo": modelo,
        "commit": _commit(),
        "entrada": entrada,
        "hiperparametros": json.dumps(hiperparametros, ensure_ascii=False, default=str),
        "particiones": len(f1_folds),
        "macro_f1_media": round(float(np.mean(f1_folds)), 4),
        "macro_f1_desvio": round(float(np.std(f1_folds)), 4),
        "segundos": round(segundos),
        "notas": notas,
    }])
    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    fila.to_csv(REGISTRO, mode="a", header=not REGISTRO.exists(), index=False)


if __name__ == "__main__":
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    muestra = pd.read_parquet(RUTA_MUESTRA, columns=["id", "artist", "tag", "letra"])
    if not RUTA_FOLDS.exists():
        guardar_folds(crear_folds(muestra))
        print("particiones guardadas en", RUTA_FOLDS)
    fold = cargar_folds(muestra["id"])
    y = muestra["tag"].to_numpy()
    Y = etiquetas(y)
    print(pd.crosstab(fold, y)[GENEROS].to_string())

    referencias = {
        "M0_mayoritaria": (lambda: DummyClassifier(strategy="most_frequent"), {"strategy": "most_frequent"}),
        "M1_tfidf_lr": (
            lambda: make_pipeline(TfidfVectorizer(**TFIDF_REFERENCIA), LogisticRegression(max_iter=1000, C=4)),
            {"tfidf": TFIDF_REFERENCIA, "C": 4, "max_iter": 1000},
        ),
    }
    for nombre, (fabrica, hiper) in referencias.items():
        t0 = time.time()
        proba, f1s = cv_oof(con_sklearn(fabrica, muestra["letra"], y), Y, fold, nombre=nombre)
        guardar_oof(nombre, muestra["id"], fold, proba)
        registrar(nombre, "letra", hiper, f1s, time.time() - t0)
