# TP de NLP — Clasificación de género musical a partir de letras

Trabajo práctico grupal de Procesamiento de Lenguaje Natural (ITBA, 2026Q2, docentes Caravaggio y Valentini).
Integrantes: Joacim Thörnquist, Alicia Cobo Iglesias, Santiago Minoyetti.
Repositorio: https://github.com/JoacimThornquist/nlp-genre-lyrics (dueño: Joacim).

El tema anterior (opinión vs. información en diarios argentinos) **se descartó por completo**. Su código quedó
fuera del repo, en `../_archivo_tp_noticias/`. No reutilizar nada de ahí ni mencionarlo en el informe.

## Objetivo (borrador, a cerrar en la 1ra entrega)

Predecir el género musical (`tag`) de una canción a partir de su letra (`lyrics`), y entender qué señales
lingüísticas distinguen a cada género. La pregunta y las hipótesis se definen en la sección
"Propuesta de análisis" del informe.

## Entregas

Consignas completas y checklist en `docs/consignas_entregas.md`. Resumen:

| Entrega | Clase | Mail hasta | Qué se entrega |
|---|---|---|---|
| 1ra | lun 28-sep-2026 | dom 27-sep 23:59 | Informe (≤3 carillas): Título, Resumen, Datos, Análisis exploratorio, Propuesta de análisis |
| 2da | lun 26-oct-2026 | dom 25-oct 23:59 | Mismo doc + sección 6 "Experimentos" (≤1 página) |
| Final | lun 16-nov-2026 | dom 15-nov 23:59 | Presentación ≤15 min + link a slides y repo. Nota individual |

El informe vive en un **Google Doc** (template de la cátedra, Arial 11, interlineado 1.15). `NLP Entrega.docx`
en esta carpeta es sólo una copia local de trabajo y no se versiona. Cada figura o tabla que vaya al informe
tiene que salir de un notebook o script de este repo.

## Datos

- Fuente: Kaggle, "5 Million Song Lyrics" v3 (letras de Genius). Archivo: `data/ds2.csv.zip`
  (3,3 GB comprimido; adentro un único `ds2.csv` de 9,2 GB, ~5,9 M filas). **No está en git.**
- Columnas: `title, tag, artist, year, views, features, lyrics, id`. Se usan `lyrics` (input),
  `tag` (target) y `artist` (para particionar).
- Géneros en el total: pop 2.519.256 · rap 1.962.010 · rock 892.220 · rb 225.342 · misc 208.714 · country 105.869.
- `misc` se excluye: no son canciones (poemas, textos bíblicos, spoken word).
- Muestra de trabajo actual (hecha en Colab): 10.000 por género × 5 géneros, semilla 42; tras quitar
  4 letras vacías y 279 duplicadas quedan 49.720 canciones de 10.832 artistas.
  Si el código de esa muestra está en Colab, **esa es la muestra canónica**: portarlo a `src/` en vez de
  reinventarlo, así el informe y el repo coinciden.
- No hay columna de idioma. Hay que filtrar por idioma antes de modelar (decisión pendiente: herramienta y umbral).

## Reglas del proyecto

1. **Nunca** commitear datos ni descomprimir el zip dentro de OneDrive (9 GB sincronizándose). Leer siempre
   directo del zip y por partes con `src/datos.py` (`leer_por_partes`, `contar_valores`).
2. Nunca cargar el CSV entero en memoria. Las muestras derivadas van a `data/processed/` en parquet (ignorado por git).
3. **Partición agrupada por artista** (`GroupShuffleSplit` / `StratifiedGroupKFold` con `groups=artist`).
   Una partición aleatoria mete canciones del mismo artista en train y test y el modelo aprende al artista, no al género.
   Deduplicar letras **antes** de partir.
4. Semilla fija `42` en todo lo aleatorio.
5. Ojo con los encabezados de sección de Genius (`[Chorus: Opera Steve & Cam'ron]`, `[Verse 1]`): traen
   nombres de artistas (fuga directa) y marcas de estructura (Hook, Intro) que son más frecuentes en rap.
   Decidir explícitamente si se quitan (`quitar_encabezados` en `src/datos.py`) y reportarlo en el preprocesamiento.
6. `year` puede ser un confusor (la mezcla de géneros cambia con los años): revisarlo en el EDA.
7. Métrica principal: macro-F1 (los géneros están desbalanceados en el total aunque la muestra sea balanceada).
   Siempre contra un baseline simple (clase mayoritaria y TF-IDF + regresión logística) antes de modelos grandes.
8. El informe tiene 3 carillas: pocas figuras, densas y legibles. Preferir una tabla bien hecha a tres gráficos.

## Estructura

```
CLAUDE.md                     este archivo (contexto para Claude Code)
README.md                     presentación del repo para docentes
docs/consignas_entregas.md    consignas oficiales de las 3 entregas + checklist
data/                         ds2.csv.zip y derivados (ignorado por git, salvo data/README.md)
notebooks/                    notebooks numerados: 01_eda.ipynb, 02_baselines.ipynb, ...
src/datos.py                  lectura por partes, conteos, muestreo, limpieza de encabezados
reports/figures/              figuras que van al informe (PNG, nombre descriptivo)
.claude/                      configuración compartida de Claude Code y skills del proyecto
```

Material de cátedra (notebooks de referencia): `../NLP/` (repo LCaravaggio/NLP). Relevantes para este TP:
`06_clasificación`, `07_TopicModelling`, `05_embeddings`, `09_Transformers`.

## Entorno

- Windows + VS Code (extensión Claude Code y Jupyter) + Git for Windows. El equipo también usa Google Colab.
- En Colab, montar Drive y apuntar la variable de entorno `LYRICS_DATA` al zip; `src/datos.py` la respeta.
- Instalar dependencias: `pip install -r requirements.txt` (conda/miniconda disponible).

## Flujo de trabajo con git (3 personas)

- `main` siempre funciona. Cada tarea en una rama (`eda-longitudes`, `baseline-tfidf`, ...) y se integra por PR.
- Mensajes de commit en español, en imperativo y cortos ("Agrega distribución de longitudes por género").
- Antes de commitear notebooks: que corran de punta a punta y sin salidas gigantes (tablas enteras, miles de líneas).
- Claude: pedir confirmación antes de `git push`, `gh pr create` o cualquier cosa que reescriba historia.
