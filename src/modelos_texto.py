"""Modelos sobre el texto de la letra: elección de C, aporte de los rasgos de forma y reentrenamiento sin pop.

Para cada una de las cinco particiones (agrupadas por artista, `folds.parquet`):

1. **Elección de C.** Se separa del entrenamiento una validación interna con el 15% de los artistas, se prueba
   cada valor de GRILLA_C y se elige el de mayor macro-F1 en esa validación. La partición de evaluación no
   interviene. Con ese C se reentrena con todo el entrenamiento y se predice la partición de evaluación.
2. **Texto + rasgos.** La misma regresión logística, con el mismo C, recibe además los 41 rasgos de forma
   (estandarizados) como columnas adicionales. La diferencia con el modelo anterior es el aporte de los rasgos.
3. **Sin pop.** La misma regresión se entrena solo con rap, rock, R&B y country y predice todas las canciones de
   la partición de evaluación, incluidas las de pop (a las que tiene que asignarles alguno de los otros géneros).

Además corre el XGBoost de los rasgos (notebook 03) solo con la forma pura: estructura, repetición, rima,
métrica y diversidad léxica, sin ninguna lista de palabras.

Uso:  python -m src.modelos_texto     (alrededor de 30 minutos en CPU)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.datos import GENEROS, RAIZ, SEMILLA
from src.evaluacion import (K, RUTA_MUESTRA, TFIDF_REFERENCIA, cargar_folds, cv_oof, division_interna, etiquetas,
                            guardar_oof, macro_f1, particiones, registrar)
from src.rasgos import GRUPOS, LINGUISTICOS, SALIDA as RUTA_RASGOS

TAB = RAIZ / "reports" / "tablas"
GRILLA_C = [0.5, 1, 2, 4, 8, 16]
POP = GENEROS.index("pop")
FORMA_PURA = [c for g in "ERSMD" for c in GRUPOS[g]]
PARAMS_XGB = dict(objective="multi:softprob", tree_method="hist", max_depth=6, learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, eval_metric="mlogloss", n_jobs=-1, random_state=SEMILLA)
MAX_ARBOLES, PARADA = 2000, 50


def _regresion(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=1000)


def _proba(modelo, X) -> np.ndarray:
    """Probabilidades en el orden de GENEROS; 0 en los géneros que el modelo no conoce."""
    p = np.zeros((X.shape[0], K))
    p[:, modelo.classes_] = modelo.predict_proba(X)
    return p


def estandarizar(R: np.ndarray, i_tr: np.ndarray) -> np.ndarray:
    """Rasgos con media 0 y desvío 1 según el entrenamiento; los faltantes toman la mediana del entrenamiento."""
    mediana = np.nanmedian(R[i_tr], axis=0)
    Z = np.where(np.isnan(R), mediana, R)
    media, desvio = Z[i_tr].mean(0), Z[i_tr].std(0)
    return np.clip((Z - media) / np.where(desvio > 0, desvio, 1), -5, 5)


def una_particion(k: int, i_tr: np.ndarray, i_te: np.ndarray, letras: np.ndarray, Y: np.ndarray,
                  artistas: np.ndarray, R: np.ndarray) -> dict:
    t0 = time.time()
    # 1. elección de C con la validación interna
    i_fit, i_val = division_interna(i_tr, artistas)
    v = TfidfVectorizer(**TFIDF_REFERENCIA)
    X_fit, X_val = v.fit_transform(letras[i_fit]), v.transform(letras[i_val])
    f1_val = {C: macro_f1(Y[i_val], _regresion(C).fit(X_fit, Y[i_fit]).predict(X_val)) for C in GRILLA_C}
    C = max(f1_val, key=f1_val.get)
    # reentrenamiento con todo el entrenamiento
    v = TfidfVectorizer(**TFIDF_REFERENCIA)
    X_tr, X_te = v.fit_transform(letras[i_tr]), v.transform(letras[i_te])
    p_texto = _proba(_regresion(C).fit(X_tr, Y[i_tr]), X_te)
    # 2. texto + rasgos, mismo C
    Z = estandarizar(R, i_tr)
    m = _regresion(C).fit(hstack([X_tr, csr_matrix(Z[i_tr])]).tocsr(), Y[i_tr])
    p_ambos = _proba(m, hstack([X_te, csr_matrix(Z[i_te])]).tocsr())
    # 3. sin pop, mismo C
    i_sp = i_tr[Y[i_tr] != POP]
    v4 = TfidfVectorizer(**TFIDF_REFERENCIA)
    m4 = _regresion(C).fit(v4.fit_transform(letras[i_sp]), Y[i_sp])
    p_sin_pop = _proba(m4, v4.transform(letras[i_te]))
    print(f"  partición {k}: C = {C} · texto {macro_f1(Y[i_te], p_texto.argmax(1)):.3f} · "
          f"texto + rasgos {macro_f1(Y[i_te], p_ambos.argmax(1)):.3f}  ({time.time() - t0:.0f} s)", flush=True)
    return {"k": k, "C": C, "f1_val": f1_val, "texto": p_texto, "ambos": p_ambos, "sin_pop": p_sin_pop,
            "segundos": time.time() - t0}


def xgb_forma_pura(R: pd.DataFrame, Y: np.ndarray, artistas: np.ndarray, fold: np.ndarray):
    """El XGBoost del notebook 03, con la misma configuración, solo con los grupos E, R, S, M y D."""
    import xgboost as xgb

    X = R[FORMA_PURA].to_numpy(dtype=np.float32)

    def entrenar_predecir(i_tr, i_te):
        i_fit, i_val = division_interna(i_tr, artistas)
        m = xgb.XGBClassifier(**PARAMS_XGB, n_estimators=MAX_ARBOLES, early_stopping_rounds=PARADA)
        m.fit(X[i_fit], Y[i_fit], eval_set=[(X[i_val], Y[i_val])], verbose=False)
        return m.predict_proba(X[i_te])

    return cv_oof(entrenar_predecir, Y, fold, nombre="M2 forma pura")


if __name__ == "__main__":
    muestra = pd.read_parquet(RUTA_MUESTRA, columns=["id", "artist", "tag", "letra"])
    R = pd.read_parquet(RUTA_RASGOS).set_index("id").loc[muestra["id"]].reset_index(drop=True)
    ids, artistas = muestra["id"].to_numpy(), muestra["artist"].to_numpy()
    letras, Y = muestra["letra"].to_numpy(), etiquetas(muestra["tag"])
    fold = cargar_folds(ids)
    R_ling = R[LINGUISTICOS].to_numpy(dtype=float)

    t0 = time.time()
    salidas = Parallel(n_jobs=2)(
        delayed(una_particion)(k, i_tr, i_te, letras, Y, artistas, R_ling)
        for k, (i_tr, i_te) in enumerate(particiones(fold)))
    segundos = time.time() - t0

    probas = {n: np.full((len(Y), K), np.nan) for n in ["texto", "ambos", "sin_pop"]}
    for s, (_, i_te) in zip(salidas, particiones(fold)):
        for n in probas:
            probas[n][i_te] = s[n]
    grilla = pd.DataFrame({s["k"]: s["f1_val"] for s in salidas}).T.rename_axis("partición")
    grilla.columns = [f"C = {c:g}" for c in grilla.columns]
    grilla["C elegido"] = [s["C"] for s in salidas]
    grilla.to_csv(TAB / "m1_grilla_C.csv")
    print(grilla.round(4).to_string())

    C_por_particion = {int(s["k"]): s["C"] for s in salidas}
    hiper = {"tfidf": TFIDF_REFERENCIA, "grilla_C": GRILLA_C, "C_por_particion": C_por_particion, "max_iter": 1000}
    for n, nombre, entrada, notas in [
        ("texto", "M1_tfidf_lr_grilla", "letra", "C elegido con la validación interna (15% de los artistas)"),
        ("ambos", "M1_tfidf_lr_rasgos", "letra + rasgos (41)", "misma regresión y mismo C que M1_tfidf_lr_grilla"),
        ("sin_pop", "M1_sin_pop", "letra", "entrenado sin pop; predice todas las canciones"),
    ]:
        # el modelo sin pop se mide solo sobre las canciones que no son de pop
        usar = [i_te[Y[i_te] != POP] if n == "sin_pop" else i_te for _, i_te in particiones(fold)]
        f1s = np.array([macro_f1(Y[i], probas[n][i].argmax(1)) for i in usar])
        guardar_oof(nombre, ids, fold, probas[n])
        registrar(nombre, entrada, hiper, f1s, segundos / 3, notas)
        print(f"{nombre}: macro-F1 {f1s.mean():.4f} ± {f1s.std():.4f}")

    t0 = time.time()
    P, f1s = xgb_forma_pura(R, Y, artistas, fold)
    guardar_oof("M2_xgb_forma_pura", ids, fold, P)
    registrar("M2_xgb_forma_pura", f"rasgos ({len(FORMA_PURA)})",
              {**PARAMS_XGB, "max_arboles": MAX_ARBOLES, "parada": PARADA, "columnas": FORMA_PURA}, f1s,
              time.time() - t0, "solo forma pura: grupos E, R, S, M, D")
