---
name: revisar-entrega
description: Revisa un borrador del informe o de la presentación del TP de NLP contra la consigna oficial de la entrega que corresponda (1ra, 2da o final) y devuelve qué falta, qué sobra y qué corregir antes de mandar el mail.
---

# Revisar una entrega del TP

Usar cuando pidan revisar, chequear o "dejar listo para entregar" el informe o las slides.

## Pasos

1. Leer `docs/consignas_entregas.md` y quedarse con la entrega que toca según la fecha de hoy
   (1ra: hasta 27-sep-2026; 2da: hasta 25-oct-2026; final: hasta 15-nov-2026). Si no está claro, preguntar.
2. Conseguir el texto del borrador: el Google Doc si hay conector de Drive, si no `NLP Entrega.docx`
   (convertir con pandoc o python-docx) o lo que pegue el usuario.
3. Chequear forma:
   - Secciones exactas y en orden (1ra: Título, Resumen, Datos, Análisis exploratorio, Propuesta de análisis;
     2da: + "Experimentos" ≤1 página; final: Objetivos, Metodología, Resultados, Conclusiones, Limitaciones, Anexo).
   - Largo: estimar carillas (≈ 550-600 palabras por carilla en Arial 11 / 1.15, menos si hay figuras).
     Si pasa de 3, proponer qué cortar.
   - Encabezado con los tres integrantes, sin carátula ni índice.
4. Chequear fondo (lo que la cátedra valora):
   - Pregunta de investigación concreta y respondible con los datos.
   - Datos: fuente, tamaño, formato y cómo se armó la muestra (semilla, filtros, deduplicado, idioma).
   - EDA: cada figura/tabla dice algo que motiva una decisión posterior; preprocesamiento explicitado.
   - Propuesta/Experimentos: baseline, métrica (macro-F1), partición **agrupada por artista**, y cómo se
     controla la fuga (encabezados `[Chorus: ...]`, duplicados, artistas repetidos).
   - Cada número del informe coincide con lo que produce el código del repo (verificar corriendo el notebook
     o `src/datos.py` si hace falta).
5. Chequear el envío: links (Google Doc con edición para docentes + repo accesible), mail en el thread original,
   sin adjuntos.

## Salida

Una lista corta, ordenada por gravedad: **bloqueantes** (incumplen la consigna), **importantes** (bajan nota),
**menores** (estilo). Para cada una, la corrección concreta. No reescribir el informe entero salvo que lo pidan.
