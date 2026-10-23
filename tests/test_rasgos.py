"""Pruebas de src/rasgos.py con estrofas anotadas a mano.

Uso:  python tests/test_rasgos.py     (o `pytest tests/` si está instalado)
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rasgos as r  # noqa: E402

AABB = "I saw a cat\nsitting on a mat\nthe night was long\nwe sang a song\nit was a sunny day\nwe went the other way"
ABAB = "the night is long\nI feel the rain\nI sing my song\nand feel no pain\nI know I'm wrong\nI miss the train"


def cerca(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_clave_rima():
    assert r.clave_rima(r.pronunciacion("cat")) == r.clave_rima(r.pronunciacion("mat")) == ("AE", "T")
    assert r.clave_rima(r.pronunciacion("long")) != r.clave_rima(r.pronunciacion("rain"))


def test_rima_pareada_y_alterna():
    a = r.rasgos_letra(AABB)        # cat-mat, long-song, day-way: 3 de 5 pares consecutivos riman
    assert cerca(a["s_rima_AA"], 3 / 5)
    assert cerca(a["s_rima_ABAB"], 0.0)
    b = r.rasgos_letra(ABAB)        # long-song-wrong y rain-pain-train: riman a distancia 2
    assert cerca(b["s_rima_AA"], 0.0)
    assert cerca(b["s_rima_ABAB"], 1.0)
    assert cerca(b["s_densidad_rima"], 4 / 5)    # desde el tercer verso, todos riman con uno anterior


def test_repetir_la_palabra_final_no_es_rimar():
    letra = "\n".join(["oh baby", "my baby", "sweet baby", "yeah baby", "come on baby"])
    assert math.isnan(r.rasgos_letra(letra)["s_rima_AA"])


def test_asonancia_y_multisilabica():
    basement, patience, placement = (r.pronunciacion(w) for w in ("basement", "patience", "placement"))
    assert r.clave_rima(basement) != r.clave_rima(patience)              # no es rima perfecta
    assert r.clave_asonancia(basement) == r.clave_asonancia(patience)    # pero las vocales coinciden
    assert r.clave_multisilabica(basement) == r.clave_multisilabica(placement)
    assert r.clave_multisilabica(basement) != r.clave_multisilabica(patience)


def test_apocope_y_silabas():
    assert r.pronunciacion("nothin'")[-1] == "N"     # el CMU trae la apócope con su pronunciación real
    assert "trippin" not in r._cmu()                  # si no la trae, se usa la forma completa
    assert r.pronunciacion("trippin'") == r.pronunciacion("tripping")
    assert r.silabas("beautiful") == 3
    assert r.silabas("tryna") == 2                # fuera del CMU: grupos de vocales escritas


def test_estribillo():
    versos = ["a", "b", "c", "a", "b", "d", "a", "b"]  # el bloque (a, b) aparece 3 veces
    assert cerca(r._cobertura_estribillo(versos), 2 * 3 / 8)
    assert r._cobertura_estribillo(["a", "b", "c", "d"]) == 0.0


def test_rima_interna():
    letra = "I got the cash and the stash\nmoney in the bank and the tank\nthe light is bright tonight\nhello"
    assert cerca(r.rasgos_letra(letra)["s_rima_interna"], 1.0)


def test_pronombres_y_vocalizaciones():
    a = r.rasgos_letra("I love you and you love me\nooh yeah la la la baby")
    assert cerca(a["p_yo"], 100 * 2 / 13) and cerca(a["p_tu"], 100 * 2 / 13)
    assert cerca(a["p_log_yo_tu"], 0.0)
    assert cerca(a["o_vocalizaciones"], 100 * 5 / 13)


def test_mtld():
    assert r.mtld(["la"] * 60) < 3                  # una sola palabra repetida: diversidad mínima
    assert math.isnan(r.mtld(["la"] * 10))          # textos cortos: sin valor


def test_transcripcion():
    a = r.rasgos_letra("HEY you (yeah)\nhello, world!\nI LOVE it")
    assert cerca(a["t_pct_parentesis"], 100 / 3)
    assert cerca(a["t_comas_por_verso"], 1 / 3)
    assert cerca(a["t_pct_mayusculas"], 100 * 2 / 8)   # HEY y LOVE ("I" tiene una sola letra)


def test_columnas():
    esperadas = {c for cols in r.GRUPOS.values() for c in cols} - {
        "x_sustantivos", "x_verbos", "x_adjetivos", "x_adverbios", "x_pasado"}
    assert set(r.rasgos_letra(AABB)) == esperadas
    assert set(r.rasgos_letra("")) == esperadas


if __name__ == "__main__":
    pruebas = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    fallas = 0
    for prueba in pruebas:
        try:
            prueba()
            print(f"ok     {prueba.__name__}")
        except AssertionError as e:
            fallas += 1
            print(f"FALLA  {prueba.__name__} {e}")
    print(f"{len(pruebas) - fallas}/{len(pruebas)} pruebas correctas")
    sys.exit(1 if fallas else 0)
