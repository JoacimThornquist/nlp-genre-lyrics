"""Resultados finales: junta las predicciones fuera de muestra de todos los modelos y arma las tablas y figuras.

Lee `data/processed/oof/*.parquet` (una fila por canción, predicha por un modelo que no vio ni la canción ni a su
artista) y, si existen, las salidas del notebook 04 (`data/processed/salida_minilm.zip`).

Pasos:
1. Embeddings congelados de MiniLM + regresión logística, con C elegido por validación interna.
2. Tabla de modelos: macro-F1 y F1 por género con IC 95% por bootstrap de artistas, y diferencia apareada contra
   el modelo de texto (TF-IDF + regresión logística).
3. Aporte de los rasgos de forma sobre el texto, por género.
4. Matriz de confusión del modelo de texto.
5. Reentrenamiento sin pop: cómo cambian los otros géneros y adónde van las canciones de pop.
6. Pureza de vecinos: qué fracción de los 20 vecinos más cercanos de cada canción es de su mismo género.
7. Desempeño por época.

Uso:
    python -m src.resultados --exportar    letras y particiones en un solo archivo para el notebook 04
    python -m src.resultados               tablas (reports/tablas), figuras (reports/figures) y resultados.json
"""
from __future__ import annotations

import json
import sys
import time
import zipfile

import numpy as np
import pandas as pd

from src.datos import GENEROS, RAIZ, SEMILLA
from src.evaluacion import (DIR_OOF, K, PROC, REGISTRO, RUTA_MUESTRA, bootstrap_artistas, cargar_folds, cargar_oof,
                            division_interna, etiquetas, f1_clases, guardar_oof, ic, macro_f1, particiones,
                            predecir, registrar)

TAB = RAIZ / "reports" / "tablas"
FIG = RAIZ / "reports" / "figures"
ZIP_MINILM = PROC / "salida_minilm.zip"
RUTA_EMB = PROC / "emb_minilm_congelado.npz"
RUTA_LDA = PROC / "lda_theta_k12.parquet"
NOMBRE = {"pop": "Pop", "rap": "Rap", "rock": "Rock", "rb": "R&B", "country": "Country"}
COLOR = {"pop": "#2a78d6", "rap": "#eb6834", "rock": "#1baf7a", "rb": "#eda100", "country": "#e87ba4"}
POP = GENEROS.index("pop")
VECINOS = 20
GRILLA_C_EMB = [1, 4, 16, 64, 256]
REFERENCIA = "M1_tfidf_lr_grilla"
# nombre del archivo de predicciones -> nombre para mostrar; el orden es el de la tabla
MODELOS = {
    "M0_mayoritaria": "Clase mayoritaria",
    "M2_xgb_forma_pura": "Forma pura (24 rasgos) + XGBoost",
    "M2_xgb_rasgos": "Rasgos de forma y estilo (41) + XGBoost",
    "M4_minilm_lr": "MiniLM congelado + regresión logística",
    "M1_tfidf_lr_grilla": "TF-IDF + regresión logística",
    "M1_tfidf_lr_rasgos": "TF-IDF + rasgos, misma regresión",
    "M5_minilm_afinado": "MiniLM con ajuste fino",
}


def exportar() -> None:
    """Letras, etiquetas, artistas y particiones en un solo archivo, para correr el notebook 04 en Colab."""
    m = pd.read_parquet(RUTA_MUESTRA, columns=["id", "artist", "tag", "letra"])
    m["fold"] = cargar_folds(m["id"])
    m.to_parquet(PROC / "letras_colab.parquet", index=False, compression="zstd")
    print(f"{len(m):,} canciones en {PROC / 'letras_colab.parquet'}")


# ----------------------------------------------------------------------------- salidas del notebook 04
def importar_minilm() -> None:
    """Descomprime salida_minilm.zip: predicciones del ajuste fino a oof/ y embeddings a data/processed/."""
    if not ZIP_MINILM.exists():
        return
    with zipfile.ZipFile(ZIP_MINILM) as z:
        z.extract("emb_minilm_congelado.npz", PROC)
        z.extract("M5_minilm_afinado.json", PROC)
        z.extract("M5_minilm_afinado.parquet", DIR_OOF)
    resumen = json.loads((PROC / "M5_minilm_afinado.json").read_text())
    ya = REGISTRO.exists() and "M5_minilm_afinado" in set(pd.read_csv(REGISTRO)["modelo"])
    if not ya:
        partes = resumen.pop("particiones")
        registrar("M5_minilm_afinado", f"letra (primeros {resumen['max_tokens']} tokens)",
                  {**resumen, "epoca_por_particion": {p["particion"]: p["epoca_elegida"] for p in partes}},
                  np.array([p["macro_f1"] for p in partes]), sum(p["segundos"] for p in partes),
                  "notebook 04; la época se elige con la validación interna")


def cargar_embeddings(ids: np.ndarray) -> np.ndarray | None:
    if not RUTA_EMB.exists():
        return None
    z = np.load(RUTA_EMB)
    E = pd.DataFrame(z["emb"].astype(np.float32), index=z["id"]).loc[ids].to_numpy()
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def regresion_embeddings(E, Y, artistas, fold, ids) -> None:
    """M4: regresión logística sobre los embeddings congelados; C se elige con la validación interna."""
    from sklearn.linear_model import LogisticRegression

    t0, proba, f1s, filas = time.time(), np.full((len(Y), K), np.nan), [], []
    for k, (i_tr, i_te) in enumerate(particiones(fold)):
        i_fit, i_val = division_interna(i_tr, artistas)
        f1_val = {C: macro_f1(Y[i_val], LogisticRegression(C=C, max_iter=2000).fit(E[i_fit], Y[i_fit]).predict(E[i_val]))
                  for C in GRILLA_C_EMB}
        C = max(f1_val, key=f1_val.get)
        proba[i_te] = LogisticRegression(C=C, max_iter=2000).fit(E[i_tr], Y[i_tr]).predict_proba(E[i_te])
        f1s.append(macro_f1(Y[i_te], proba[i_te].argmax(1)))
        filas.append({"partición": k, **{f"C = {c:g}": v for c, v in f1_val.items()}, "C elegido": C})
        print(f"  MiniLM congelado, partición {k}: C = {C}, macro-F1 {f1s[-1]:.3f}", flush=True)
    pd.DataFrame(filas).set_index("partición").to_csv(TAB / "m4_grilla_C.csv")
    guardar_oof("M4_minilm_lr", ids, fold, proba)
    registrar("M4_minilm_lr", "embeddings all-MiniLM-L6-v2 (384), primeros 256 tokens",
              {"grilla_C": GRILLA_C_EMB, "C_por_particion": {f["partición"]: f["C elegido"] for f in filas},
               "max_iter": 2000}, np.array(f1s), time.time() - t0, "encoder congelado; notebook 04")


# ----------------------------------------------------------------------------- comparaciones
def tabla_modelos(probas: dict, Y, artistas) -> pd.DataFrame:
    """Una fila por modelo, sobre las canciones que predijo, con la diferencia apareada contra REFERENCIA."""
    filas = []
    for archivo, P in probas.items():
        comunes = ~np.isnan(P).all(1) & ~np.isnan(probas[REFERENCIA]).all(1)
        preds = {"m": predecir(P[comunes]), "ref": predecir(probas[REFERENCIA][comunes])}
        boot = bootstrap_artistas(preds, Y[comunes], artistas[comunes])
        f = f1_clases(Y[comunes], preds["m"])
        lo, hi = ic(boot["m"][:, K])
        fila = {"modelo": MODELOS[archivo], "archivo": archivo, "canciones": int(comunes.sum()),
                "macro-F1": f.mean(), "IC inf": lo, "IC sup": hi, **{NOMBRE[g]: f[i] for i, g in enumerate(GENEROS)}}
        if archivo != REFERENCIA:
            d = boot["m"][:, K] - boot["ref"][:, K]
            fila["Δ vs texto"] = f.mean() - f1_clases(Y[comunes], preds["ref"]).mean()
            fila["Δ IC inf"], fila["Δ IC sup"] = ic(d)
        filas.append(fila)
    return pd.DataFrame(filas).set_index("modelo")


def diferencia_por_genero(P_a, P_b, Y, artistas, nombres=("F1 sin rasgos", "F1 con rasgos")) -> pd.DataFrame:
    """F1 de b menos F1 de a, por género y en macro, con IC 95% apareado, sobre las canciones que predijeron ambos."""
    comunes = ~np.isnan(P_a).all(1) & ~np.isnan(P_b).all(1)
    Y, artistas = Y[comunes], artistas[comunes]
    preds = {"a": predecir(P_a[comunes]), "b": predecir(P_b[comunes])}
    boot = bootstrap_artistas(preds, Y, artistas)
    d = boot["b"] - boot["a"]
    lo, hi = ic(d)
    fa, fb = f1_clases(Y, preds["a"]), f1_clases(Y, preds["b"])
    return pd.DataFrame({nombres[0]: np.append(fa, fa.mean()), nombres[1]: np.append(fb, fb.mean()),
                         "Δ F1": np.append(fb - fa, fb.mean() - fa.mean()), "IC inf": lo, "IC sup": hi,
                         "excluye 0": (lo > 0) | (hi < 0)}, index=[NOMBRE[g] for g in GENEROS] + ["macro-F1"])


def confusion(Y, pred) -> pd.DataFrame:
    """Matriz de confusión en cantidades: filas = género real, columnas = género predicho."""
    cm = np.bincount(Y * K + pred, minlength=K * K).reshape(K, K)
    return pd.DataFrame(cm, index=[NOMBRE[g] for g in GENEROS], columns=[NOMBRE[g] for g in GENEROS])


def sin_pop(P_con, P_sin, Y, artistas) -> tuple[pd.DataFrame, pd.Series]:
    """Efecto de sacar el pop del entrenamiento, medido sobre las canciones que no son de pop.

    Con pop: el modelo de cinco géneros (una predicción "pop" cuenta como error). Sin pop: el de cuatro géneros.
    Devuelve además a qué género manda el modelo sin pop a las canciones de pop.
    """
    otras = Y != POP
    preds = {"con": predecir(P_con[otras]), "sin": predecir(P_sin[otras])}
    boot = bootstrap_artistas(preds, Y[otras], artistas[otras])
    resto = [k for k in range(K) if k != POP]
    f_con, f_sin = f1_clases(Y[otras], preds["con"])[resto], f1_clases(Y[otras], preds["sin"])[resto]
    d = np.c_[(boot["sin"] - boot["con"])[:, resto], (boot["sin"] - boot["con"])[:, resto].mean(1)]
    lo, hi = ic(d)
    recall_con = np.array([(preds["con"][Y[otras] == k] == k).mean() for k in resto])
    recall_sin = np.array([(preds["sin"][Y[otras] == k] == k).mean() for k in resto])
    a_pop = np.array([(preds["con"][Y[otras] == k] == POP).mean() for k in resto])
    tabla = pd.DataFrame({"F1 con pop": np.append(f_con, f_con.mean()), "F1 sin pop": np.append(f_sin, f_sin.mean()),
                          "Δ F1": np.append(f_sin - f_con, f_sin.mean() - f_con.mean()), "IC inf": lo, "IC sup": hi,
                          "recall con pop": np.append(recall_con, recall_con.mean()),
                          "recall sin pop": np.append(recall_sin, recall_sin.mean()),
                          "% predicho como pop": np.append(a_pop, a_pop.mean())},
                         index=[NOMBRE[GENEROS[k]] for k in resto] + ["promedio"])
    destino = pd.Series(np.bincount(predecir(P_sin[~otras]), minlength=K)[resto] / (~otras).sum(),
                        index=[NOMBRE[GENEROS[k]] for k in resto], name="destino de las canciones de pop")
    return tabla, destino


def pureza_vecinos(X, Y, artistas, n_vecinos: int = VECINOS, bloque: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Para cada canción, géneros de sus `n_vecinos` canciones más parecidas (coseno), sin contar las de su artista.

    Devuelve la pureza por género (fracción de vecinos del mismo género, con IC 95% por bootstrap de artistas;
    con cinco géneros balanceados el azar da 0,20) y la composición media de los vecinos de cada género.
    """
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    cod_artista = pd.factorize(artistas)[0]
    extra = 60                                         # margen para descartar canciones del mismo artista
    comp = np.zeros((len(Y), K))
    for a in range(0, len(Y), bloque):
        sim = X[a:a + bloque] @ X.T
        cerca = np.argpartition(-sim, n_vecinos + extra, axis=1)[:, :n_vecinos + extra]
        orden = np.take_along_axis(sim, cerca, 1).argsort(1)[:, ::-1]
        cerca = np.take_along_axis(cerca, orden, 1)
        for j in range(cerca.shape[0]):
            v = cerca[j][cod_artista[cerca[j]] != cod_artista[a + j]][:n_vecinos]
            comp[a + j] = np.bincount(Y[v], minlength=K) / len(v)
    propia = comp[np.arange(len(Y)), Y]
    rng = np.random.default_rng(SEMILLA)
    n_art = cod_artista.max() + 1
    boot = np.zeros((1000, K))
    for b in range(1000):
        w = rng.multinomial(n_art, np.full(n_art, 1 / n_art))[cod_artista]
        boot[b] = np.bincount(Y, weights=w * propia, minlength=K) / np.bincount(Y, weights=w, minlength=K)
    lo, hi = ic(boot)
    pureza = pd.DataFrame({"pureza": [propia[Y == k].mean() for k in range(K)], "IC inf": lo, "IC sup": hi},
                          index=[NOMBRE[g] for g in GENEROS])
    composicion = pd.DataFrame([comp[Y == k].mean(0) for k in range(K)], index=[NOMBRE[g] for g in GENEROS],
                               columns=[NOMBRE[g] for g in GENEROS])
    return pureza, composicion


def por_epoca(probas: dict, Y, anio) -> pd.DataFrame:
    """Macro-F1 de cada modelo en tres períodos (los géneros presentes en cada período pesan lo mismo)."""
    epoca = pd.cut(pd.Series(anio), [0, 2009, 2014, 2022], labels=["hasta 2009", "2010–2014", "2015–2022"])
    filas = {}
    for archivo, P in probas.items():
        ok = ~np.isnan(P).all(1)
        filas[MODELOS[archivo]] = {e: macro_f1(Y[ok & (epoca == e).to_numpy()], predecir(P[ok & (epoca == e).to_numpy()]))
                                   for e in epoca.cat.categories}
    tabla = pd.DataFrame(filas).T
    tabla.loc["canciones"] = epoca.value_counts().reindex(epoca.cat.categories).to_numpy()
    return tabla


# ----------------------------------------------------------------------------- figuras
def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.edgecolor": "#52514e", "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "font.family": "Arial",
    })
    return plt


def figuras(tabla, aporte, cm, t_sin_pop, destino, vecinos) -> None:
    plt = _estilo()
    colores = [COLOR[g] for g in GENEROS]

    # 1. modelos
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(tabla) + 0.9))
    y = np.arange(len(tabla))[::-1]
    ax.barh(y, tabla["macro-F1"], color=["#2a78d6" if a == REFERENCIA else "#9b9a94" for a in tabla["archivo"]],
            xerr=[tabla["macro-F1"] - tabla["IC inf"], tabla["IC sup"] - tabla["macro-F1"]],
            error_kw=dict(ecolor="#0b0b0b", capsize=2, lw=0.8))
    for yi, v in zip(y, tabla["macro-F1"]):
        ax.text(v + 0.012, yi, f"{v:.3f}".replace(".", ","), va="center", fontsize=8.5)
    ax.set_yticks(y, tabla.index)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("macro-F1 (IC 95% por bootstrap de artistas)")
    ax.set_title("Desempeño de cada modelo")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "resultados_modelos.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 2. aporte de los rasgos por género
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    a = aporte.iloc[:K]
    ax.bar(a.index, a["Δ F1"], color=colores, yerr=[a["Δ F1"] - a["IC inf"], a["IC sup"] - a["Δ F1"]],
           error_kw=dict(ecolor="#0b0b0b", capsize=3, lw=0.8))
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_ylabel("Δ F1 al sumar los rasgos (IC 95%)")
    ax.set_title("Aporte de los rasgos de forma sobre el texto")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "rasgos_aporte_regresion.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 3. matriz de confusión (porcentaje de cada género real)
    pct = cm.div(cm.sum(1), axis=0).to_numpy() * 100
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=80)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if pct[i, j] > 45 else "#0b0b0b", fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(K), cm.columns)
    ax.set_yticks(range(K), cm.index)
    ax.set_xlabel("género predicho")
    ax.set_ylabel("género real")
    ax.set_title("Matriz de confusión del modelo de texto (% por fila)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(FIG / "confusion_texto.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 4. sin pop
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 2.9), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.35))
    t = t_sin_pop.iloc[:-1]
    x = np.arange(len(t))
    a1.bar(x - 0.2, t["F1 con pop"], 0.4, color="#9b9a94", label="con pop (5 géneros)")
    a1.bar(x + 0.2, t["F1 sin pop"], 0.4, color=[COLOR[g] for g in GENEROS if g != "pop"])
    a1.bar([0], [0], color="#52514e", label="sin pop (4 géneros), en el color de cada género")
    a1.set_xticks(x, t.index)
    a1.set_ylabel("F1 sobre canciones que no son pop")
    a1.set_title("Los otros géneros, con y sin pop")
    a1.set_ylim(0, 1.05)
    a1.legend(fontsize=8, loc="upper left", ncol=1)
    a1.grid(axis="x", visible=False)
    a2.bar(destino.index, destino * 100, color=[COLOR[g] for g in GENEROS if g != "pop"])
    for i, v in enumerate(destino):
        a2.text(i, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=8.5)
    a2.set_ylabel("% de las canciones de pop")
    a2.set_title("Adónde va el pop sin su etiqueta")
    a2.grid(axis="x", visible=False)
    fig.savefig(FIG / "pop_sin_pop.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 5. pureza de vecinos
    for nombre, (pureza, comp) in vecinos.items():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.4))
        a1.bar(pureza.index, pureza["pureza"] * 100, color=colores,
               yerr=[(pureza["pureza"] - pureza["IC inf"]) * 100, (pureza["IC sup"] - pureza["pureza"]) * 100],
               error_kw=dict(ecolor="#0b0b0b", capsize=3, lw=0.8))
        a1.axhline(20, color="#52514e", ls="--", lw=0.9)
        a1.text(K - 0.5, 21, "azar (20%)", ha="right", fontsize=8, color="#52514e")
        a1.set_ylabel(f"% de los {VECINOS} vecinos del mismo género")
        a1.set_title("Pureza de vecinos")
        a1.grid(axis="x", visible=False)
        izq = np.zeros(K)
        for j, g in enumerate(GENEROS):
            a2.barh(comp.index[::-1], comp.iloc[::-1, j] * 100, left=izq, color=COLOR[g], label=NOMBRE[g])
            izq += comp.iloc[::-1, j].to_numpy() * 100
        a2.set_xlim(0, 100)
        a2.set_xlabel("género de los vecinos (%)")
        a2.set_title("De qué género son los vecinos de cada género")
        a2.legend(fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.28), handlelength=1, columnspacing=1)
        a2.grid(False)
        fig.savefig(FIG / f"pop_vecinos_{nombre}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)


# ----------------------------------------------------------------------------- principal
def principal() -> None:
    muestra = pd.read_parquet(RUTA_MUESTRA, columns=["id", "artist", "tag", "year"])
    ids, artistas, Y = muestra["id"].to_numpy(), muestra["artist"].to_numpy(), etiquetas(muestra["tag"])
    fold = cargar_folds(ids)

    importar_minilm()
    E = cargar_embeddings(ids)
    if E is not None and not (DIR_OOF / "M4_minilm_lr.parquet").exists():
        regresion_embeddings(E, Y, artistas, fold, ids)

    probas = {a: cargar_oof(a, ids) for a in MODELOS if (DIR_OOF / f"{a}.parquet").exists()}
    tabla = tabla_modelos(probas, Y, artistas)
    tabla.to_csv(TAB / "resultados_modelos.csv")
    print(tabla.drop(columns="archivo").round(3).to_string(), "\n")

    aporte = diferencia_por_genero(probas[REFERENCIA], probas["M1_tfidf_lr_rasgos"], Y, artistas)
    aporte.to_csv(TAB / "rasgos_aporte_regresion.csv")
    print(aporte.round(3).to_string(), "\n")

    ajuste = None
    if "M4_minilm_lr" in probas and "M5_minilm_afinado" in probas:
        ajuste = diferencia_por_genero(probas["M4_minilm_lr"], probas["M5_minilm_afinado"], Y, artistas,
                                       nombres=("F1 congelado", "F1 con ajuste fino"))
        ajuste.to_csv(TAB / "minilm_ajuste_vs_congelado.csv")
        print(ajuste.round(3).to_string(), "\n")

    cm = confusion(Y, predecir(probas[REFERENCIA]))
    cm.to_csv(TAB / "confusion_texto.csv")
    print(cm.div(cm.sum(1), axis=0).round(3).to_string(), "\n")

    t_sin_pop, destino = sin_pop(probas[REFERENCIA], cargar_oof("M1_sin_pop", ids), Y, artistas)
    t_sin_pop.to_csv(TAB / "pop_sin_pop.csv")
    destino.to_csv(TAB / "pop_destino_sin_pop.csv")
    print(t_sin_pop.round(3).to_string(), "\n", destino.round(3).to_string(), "\n")

    vecinos = {}
    if E is not None:
        vecinos["embeddings"] = pureza_vecinos(E, Y, artistas)
    if RUTA_LDA.exists():
        theta = pd.read_parquet(RUTA_LDA).set_index("id").loc[ids].to_numpy(dtype=np.float32)
        vecinos["topicos"] = pureza_vecinos(np.sqrt(theta), Y, artistas)   # raíz: coseno = afinidad de Bhattacharyya
    for nombre, (pureza, comp) in vecinos.items():
        pureza.to_csv(TAB / f"pop_vecinos_pureza_{nombre}.csv")
        comp.to_csv(TAB / f"pop_vecinos_composicion_{nombre}.csv")
        print(nombre, "\n", pureza.round(3).to_string(), "\n", comp.round(3).to_string(), "\n")

    epocas = por_epoca(probas, Y, muestra["year"].to_numpy())
    epocas.to_csv(TAB / "resultados_por_epoca.csv")
    print(epocas.round(3).to_string())

    figuras(tabla, aporte, cm, t_sin_pop, destino, vecinos)

    resumen = {
        "modelos": json.loads(tabla.reset_index().to_json(orient="records")),
        "aporte_rasgos": json.loads(aporte.reset_index(names="genero").to_json(orient="records")),
        "confusion": {"generos": list(cm.index), "cantidades": cm.to_numpy().tolist()},
        "sin_pop": json.loads(t_sin_pop.reset_index(names="genero").to_json(orient="records")),
        "destino_pop": destino.to_dict(),
        "vecinos": {n: {"pureza": json.loads(p.reset_index(names="genero").to_json(orient="records")),
                        "composicion": c.round(4).to_dict(orient="index")} for n, (p, c) in vecinos.items()},
        "por_epoca": epocas.to_dict(orient="index"),
    }
    if ajuste is not None:
        resumen["ajuste_vs_congelado"] = json.loads(ajuste.reset_index(names="genero").to_json(orient="records"))
    for nombre in ["m1_grilla_C", "m4_grilla_C"]:
        if (TAB / f"{nombre}.csv").exists():
            resumen[nombre] = json.loads(pd.read_csv(TAB / f"{nombre}.csv").to_json(orient="records"))
    if (PROC / "M5_minilm_afinado.json").exists():
        resumen["ajuste_fino"] = json.loads((PROC / "M5_minilm_afinado.json").read_text())
    (TAB / "resultados.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nlisto: reports/tablas/resultados.json")


if __name__ == "__main__":
    exportar() if "--exportar" in sys.argv else principal()
