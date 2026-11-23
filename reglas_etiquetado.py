"""
reglas_etiquetado.py
====================
Etiquetado SEMIAUTOMÁTICO de reseñas de Amazon (en inglés) en tres clases:

    - "Envío"       -> retrasos, daños durante el transporte, paquetes perdidos,
                       artículo equivocado o faltante en la entrega.
    - "Producto"    -> defectos, fallas o problemas del propio producto.
    - "Sentimiento" -> satisfacción o insatisfacción general, sin un problema
                       concreto de producto ni de envío.

¿Por qué reglas y no etiquetado 100 % manual?
    Leer y etiquetar 15,000 reseñas a mano no es viable para el equipo. Las reglas
    generan una primera etiqueta ("etiqueta_reglas") que después se valida con una
    muestra etiquetada a mano (datos/muestra_validacion.csv).

Prioridad cuando una reseña cumple varias reglas:  Envío > Producto > Sentimiento
    1. Envío primero: es la clase más rara y la más "accionable" para logística.
       Además, cuando un artículo llega roto por el transporte, la reseña casi siempre
       usa también palabras de producto ("broken", "cracked"); si Producto ganara,
       nunca se detectarían los daños de transporte.
    2. Producto después: un defecto concreto es más informativo que una opinión
       general.
    3. Sentimiento es la clase "por defecto": la reseña opina, pero no describe un
       problema específico de producto o de envío.

Los LÉXICOS están en listas de Python para que sea fácil editarlos: agrega o quita
expresiones regulares y vuelve a correr el notebook 02.
"""

import re

# ---------------------------------------------------------------------------
# 1. Negaciones: si una palabra de problema va precedida (a 5 palabras o menos)
#    por una negación, NO cuenta como problema ("no leaks", "not damaged",
#    "without any scratches", "hasn't broken").
# ---------------------------------------------------------------------------
NEGACIONES = r"(?:not|no|never|without|zero|nothing|none|hardly|barely|isn't|wasn't|weren't|aren't|didn't|doesn't|don't|hasn't|haven't|hadn't|won't|wouldn't|nor|free of|free from)"
VENTANA_NEGACION = 5  # número de palabras previas que se revisan


def _negado(texto, inicio):
    """True si alguna de las VENTANA_NEGACION palabras previas a 'inicio' es una negación."""
    previo = texto[max(0, inicio - 80):inicio]
    palabras = re.findall(r"[a-z']+", previo)[-VENTANA_NEGACION:]
    return any(re.fullmatch(NEGACIONES, p) for p in palabras) or bool(
        re.search(r"free (of|from)\s*$", previo)
    )


# ---------------------------------------------------------------------------
# 2. Léxico de ENVÍO
# ---------------------------------------------------------------------------
# 2a. Frases FUERTES: por sí solas ya describen un problema de envío/entrega.
ENVIO_FUERTE = [
    r"never (arrived|came|showed up|got here|was delivered|received)",
    r"(did not|didn't|has not|hasn't|have not|haven't|never) (receive|received) (it|my|the|this|any|a|an|one|all)\b",
    r"(did not|didn't|has not|hasn't|still hasn't|still has not|never) (arrive|arrived)",
    r"not (yet )?(been )?(delivered|received)",
    r"never (got|received) (it|my (order|package|item)|the (order|package|item|product)|this (order|package|item))\b(?! back)",
    r"(arrived|came|delivered|received it|got it|showed up|was shipped|shipped) (\w+ ){0,2}(late|days late|weeks late|a week late|after the (delivery|estimated|expected) date)",
    r"late (delivery|shipment|arrival)",
    r"(delivery|shipping|shipment|package|order) (\w+ ){0,3}(was |were |got |is )?(delayed|late|lost|stolen)",
    r"\bdelay(ed|s)? (in |with )?(the )?(delivery|shipping|shipment|arrival)",
    r"(shipping|delivery|shipment) (\w+ ){0,2}(took|takes|was taking) (forever|too long|so long|a long time|weeks|over a week|a month|\d+ (days|weeks))",
    r"took (forever|too long|so long|weeks|over (a|two|three) weeks?|a month|over a month|\d+ (days|weeks)|(two|three|four|five|six|several) (weeks|days)) to (arrive|ship|be delivered|get here|come|receive)",
    r"(slow|terrible|horrible|awful|poor|bad|lousy) (shipping|delivery|shipper|packaging|packing)",
    r"(shipping|delivery) (was|is) (slow|terrible|horrible|awful|poor|bad|a nightmare)",
    r"(arrived|came|was delivered|delivered|received( it)?|got( it)?|showed up|came in|arrived in) (\w+ ){0,2}(damaged|broken|broke|cracked|shattered|smashed|dented|crushed|bent|in pieces|chipped|scratched|destroyed|leaking|opened|torn|ripped)",
    r"(damaged|broken|broke|cracked|shattered|smashed|dented|crushed|destroyed) (\w+ ){0,2}(in|during|from) (shipping|shipment|transit|transport|delivery|the mail|the post)",
    r"(shipping|transit|delivery) damage",
    r"(box|package|packaging|carton|shipping box|outer box) (\w+ ){0,3}(was |were |is |came |arrived )(\w+ ){0,2}(damaged|crushed|smashed|torn|ripped|open|opened|wet|destroyed|dented|beat up|beaten up|busted)",
    r"(crushed|damaged|smashed|torn|ripped|opened|wet|destroyed|dented|beat up) (box|package|packaging|carton)",
    r"(poorly|badly|improperly|not well|inadequately|carelessly) (packaged|packed|wrapped|boxed)",
    r"(no|zero|not enough|insufficient|inadequate) (bubble wrap|packing material|packing paper|padding in the box|protection in the box)",
    r"(wasn't|was not|weren't|were not|not) (packed|packaged|wrapped|boxed) (very |that |too )?(well|properly|securely|carefully)",
    r"(received|sent|got|shipped|delivered|was sent|were sent|came with) (me )?(the |a |an )?(completely )?wrong (item|product|color|colour|size|one|model|thing|part|lid|style|pattern|order|set|version)",
    r"wrong (item|product|order) (was )?(sent|shipped|delivered|received)",
    r"not what i ordered",
    r"(item|items|package|order|piece|pieces|part|parts|product) (was |were )?missing from (the |my )?(package|order|box|shipment|delivery)",
    r"missing (item|items|package|order)\b",
    r"(only|just) (received|got) (one|two|1|2|half|part) (of|out of)",
    r"(lost|stolen) (in|by|during) (the )?(mail|transit|shipping|delivery|post|ups|fedex|usps|carrier|courier)",
    r"(package|order|item|shipment) (was |got )?(lost|stolen)",
    r"(ups|fedex|usps|dhl|amazon logistics|the carrier|the courier|the driver|the mailman|the delivery (guy|person|driver)) (\w+ ){0,4}(lost|damaged|broke|crushed|threw|left|dropped|never|late|delayed|delivered (it )?to the wrong)",
    r"delivered to the wrong (address|house|place|person)",
    r"(tracking|tracking number) (\w+ ){0,3}(never|not|wrong|didn't|doesn't|did not)",
    r"still waiting (for|on) (it|my|the|this)",
    r"(used|returned|previously opened|open box|refurbished) (item|product|one) (was )?(sent|shipped|received)",
    r"(sent|shipped|received|got) (me )?(a |an )?(used|returned|previously (used|opened|returned))",
]

# 2b. Términos GENÉRICOS de envío: solo cuentan si en la MISMA oración hay un
#     contexto negativo. Así "fast shipping" o "arrived quickly" NO se etiquetan.
# Términos claramente logísticos: aceptan un contexto negativo amplio.
ENVIO_TERMINO_FUERTE = r"\b(shipping|shipment|delivery|courier|carrier|ups|fedex|usps|dhl|mailman|tracking|packaging|shipper)\b"
ENVIO_CONTEXTO_AMPLIO = r"\b(late|delay|delayed|slow|lost|stolen|damaged|damage|crushed|smashed|dented|missing|wrong|terrible|horrible|awful|poor|poorly|forever|nightmare|problem|problems|issue|issues|complain\w*|unacceptable|ruined)\b"
# Términos que también se usan fuera del envío ("when it arrived I was disappointed"):
# exigen un contexto negativo más específico de logística.
ENVIO_TERMINO_DEBIL = r"\b(arrived|arrive|arrival|package|delivered|shipped|mail|transit)\b"
ENVIO_CONTEXTO_ESTRECHO = r"\b(late|delay|delayed|lost|stolen|damaged|crushed|smashed|dented|missing|wrong)\b"

# Elogios o menciones neutrales del envío: se BORRAN antes de evaluar el contexto,
# para que "fast shipping, but the lid is cheap" no se confunda con un problema.
ENVIO_POSITIVO = [
    r"(fast|quick|speedy|prompt|rapid|super fast|very fast|quickly|on time|early|ahead of schedule|timely|free|prime|great|good|excellent|fine|safely|safe|securely|perfect|perfectly|intact|well|nicely|beautifully|carefully|promptly|right away|in a timely manner|as promised|as expected)\s+(shipping|delivery|shipped|delivered|arrived|packaged|packed|wrapped|arrival)",
    r"(shipping|delivery|shipped|delivered|arrived|packaged|packed|wrapped|arrival|came)\s+(was |were |is )?(very |super |really |so |extremely )?(fast|quick|quickly|speedy|prompt|promptly|on time|early|timely|free|safely|safe|securely|well|nicely|perfectly|intact|great|good|excellent|fine|as promised|as expected|in perfect condition|in good condition|in great condition|undamaged|in one piece|without (a |any )?(problem|issue|damage|scratch))",
    r"(no|not|without any|without) (problem|problems|issue|issues|damage|delay|delays) (with|in|during) (the )?(shipping|delivery|shipment|packaging)",
]


# ---------------------------------------------------------------------------
# 3. Léxico de PRODUCTO (defectos, fallas, mala calidad, mal ajuste)
# ---------------------------------------------------------------------------
# 3a. Frases que YA incluyen su propia negación (por eso no se filtran por negación).
PRODUCTO_FRASES_NEGATIVAS = [
    r"(does not|doesn't|did not|didn't|do not|don't|will not|won't|would not|wouldn't|could not|couldn't|cannot|can't|can not) (\w+ ){0,1}(work|function|fit|stay|hold|stick|seal|close|open|lock|heat|turn on|charge|spin|cut|clean|latch|stand|lay flat|lie flat|get hot|absorb|grip)\b",
    r"\b(not|never|no longer|isn't|wasn't) (\w+ ){0,1}(working|work|worked|functioning|fitting|sealing|heating)\b",
    r"never (worked|fit|sealed)\b",
    r"(stopped|quit|ceased|stop) (working|functioning|heating|spinning|charging|sticking)",
    r"(not|no longer) (as |very )?(sturdy|durable|stable|non.?stick|waterproof|leak.?proof)",
    r"not (made )?(well|good) (made|quality)",
]

# 3b. Términos de problema SIMPLES (se descartan si están negados).
PRODUCTO_TERMINOS = [
    r"\bbroke\b", r"\bbroken\b", r"\bdamaged\b", r"\bbreaks?\b", r"\bbreaking\b", r"\bsnapped\b",
    r"\bdefect(s|ive)?\b", r"\bfaulty\b", r"\bmalfunction(s|ed|ing)?\b", r"\bdud\b",
    r"\bflimsy\b", r"\bpoor(ly)? (quality|made|constructed|designed|built|stitched|sewn)\b",
    r"\b(low|bad|terrible|horrible|awful|cheap) quality\b",
    r"\bcheaply (made|built|constructed)\b",
    r"\bcheap (plastic|material|materials|construction|metal|fabric|feel|feeling|looking|thin|flimsy|junk|knock.?off)\b",
    r"\b(feels?|felt|looks?|looked|seems?|seemed) (very |so |really |too )?cheap\b",
    r"\bleak(s|ed|ing|y)?\b", r"\bcrack(s|ed|ing)?\b", r"\bchip(s|ped|ping)\b",
    r"\brust(s|ed|ing|y)?\b", r"\bcorrod(e|ed|es|ing)\b", r"\bcorrosion\b",
    r"\b(smell|smells|smelled|smelt|odor|odour|stinks?|stank|stench|fumes)\b",
    r"\bfell apart\b", r"\bfalls? apart\b", r"\bfalling apart\b", r"\bcame apart\b", r"\bcoming apart\b",
    r"\bfell off\b", r"\bfalls? off\b", r"\bcame off\b", r"\bcomes off\b", r"\bcame loose\b",
    r"\b(tore|torn|tear(s|ing)?|ripped|rips)\b",
    r"\bpeel(s|ed|ing)?\b", r"\bflak(e|es|ed|ing)\b", r"\bwarp(s|ed|ing)?\b", r"\bbent\b",
    r"\bmelt(s|ed|ing)?\b", r"\bburn(t|ed|s)? (out|up)\b", r"\bscratch(es|ed)?\b",
    r"\bstain(s|ed)\b", r"\bfad(ed|es|ing)\b", r"\bmold(y)?\b", r"\bmould(y)?\b",
    r"\bshr(ank|unk)\b", r"\bpill(s|ed|ing)\b", r"\bunravel(ed|ing|s)?\b", r"\bfray(s|ed|ing)\b",
    r"\bwobbl(y|es|ing)\b", r"\bunstable\b", r"\buneven\b", r"\bdents? easily\b",
    r"\bwore out\b", r"\bworn out\b", r"\bwears? out\b",
    r"\bunusable\b", r"\binoperable\b", r"\bdead on arrival\b", r"\bdoa\b",
    r"\b(too|way) (small|big|large|short|long|thin|narrow|wide|tight|loose|flimsy|heavy|light|weak|hard|soft|stiff|loud|noisy|bright|dim|slippery|rough)\b",
    r"\b(very|really|extremely|super) (thin|flimsy|weak|loud|noisy|slippery|rough|hard to clean|hard to use)\b",
    r"\b(much |way |a lot |significantly )?(smaller|larger|bigger|thinner|shorter|lighter|darker|weaker) than (expected|pictured|advertised|described|shown|it looks|in the (picture|photo|photos|pictures))\b",
    r"\b(does not|doesn't|did not|didn't) look (\w+ ){0,2}like (the )?(picture|pictures|photo|photos|image|images|description)\b",
    r"\bnot (very |as |that |really )?(easy|soft|absorbent|sharp|strong|comfortable|hot|warm|cold|effective|accurate|bright|thick|durable|sturdy|stable|secure|sticky|powerful) ?(to use|to clean|enough)?\b",
    r"\b(snag(s|ged|ging)|sharp edges|blotch(es|y)|(got|get|gets|has|had|developed|have|with) (little |tiny |small |a few |some )?holes)\b",
    r"\buseless\b", r"\bdoesn't do (anything|much)\b", r"\bdoes not do (anything|much)\b",
    r"\bnoisy\b", r"\bloud\b",
    r"\bmissing (parts|pieces|screws|hardware|piece|part|instructions|components)\b",
    r"\bsticks? to (the )?(pan|pot|bottom)\b", r"\bnon.?stick (coating )?(came|comes|is coming|wore|wears|started)\b",
    r"\bdangerous\b", r"\bhazard(ous)?\b", r"\bshock(ed)?\b", r"\bsparks?\b", r"\boverheat(s|ed|ing)?\b",
    r"\bpoor(ly)? (fit|fitting|design|designed)\b", r"\bdesign flaw\b", r"\bflawed\b",
    r"\b(hard|difficult|impossible) to (assemble|use|clean|open|close|operate|install)\b",
    r"\bnot as (described|pictured|advertised|shown)\b", r"\bmisleading\b",
    r"\bfalse advertising\b", r"\bdifferent (color|colour|size) than (pictured|shown|advertised)\b",
]


# ---------------------------------------------------------------------------
# 4. Funciones de búsqueda
# ---------------------------------------------------------------------------
def _primera_coincidencia(patrones, texto, revisar_negacion):
    """Devuelve el primer patrón que coincide (y no está negado) o None."""
    for patron in patrones:
        # \b al inicio evita coincidencias dentro de otra palabra ("became" no es "came")
        for m in re.finditer(r"\b(?:" + patron + ")", texto):
            if revisar_negacion and _negado(texto, m.start()):
                continue
            return m.group(0)
    return None


def _normalizar(texto):
    """Minúsculas, sin HTML y con apóstrofes uniformes (solo para aplicar las reglas)."""
    t = str(texto).lower()
    t = re.sub(r"<[^>]+>", " ", t)           # <br />, etc.
    t = t.replace("’", "'").replace("`", "'")
    t = re.sub(r"\s+", " ", t)
    return t


def detectar_envio(texto):
    """Busca evidencia de problema de envío. Devuelve el fragmento disparador o None."""
    # (1) Frases fuertes (con revisión de negación: "did not arrive damaged" no cuenta)
    hallazgo = _primera_coincidencia(ENVIO_FUERTE, texto, revisar_negacion=True)
    if hallazgo:
        return hallazgo
    # (2) Término genérico + contexto negativo en la misma oración,
    #     después de borrar los elogios al envío.
    limpio = texto
    for patron in ENVIO_POSITIVO:
        limpio = re.sub(patron, " ", limpio)
    for oracion in re.split(r"[.!?;]+", limpio):
        for termino, contexto in ((ENVIO_TERMINO_FUERTE, ENVIO_CONTEXTO_AMPLIO),
                                  (ENVIO_TERMINO_DEBIL, ENVIO_CONTEXTO_ESTRECHO)):
            t = re.search(termino, oracion)
            if not t:
                continue
            for neg in re.finditer(contexto, oracion):
                if not _negado(oracion, neg.start()):
                    return f"{t.group(0)} ... {neg.group(0)}"
    return None


def detectar_producto(texto):
    """Busca evidencia de problema del producto. Devuelve el fragmento disparador o None."""
    hallazgo = _primera_coincidencia(PRODUCTO_FRASES_NEGATIVAS, texto, revisar_negacion=False)
    if hallazgo:
        return hallazgo
    return _primera_coincidencia(PRODUCTO_TERMINOS, texto, revisar_negacion=True)


def etiquetar_resena(texto):
    """
    Asigna UNA etiqueta a la reseña siguiendo la prioridad Envío > Producto > Sentimiento.
    Devuelve (etiqueta, regla_disparada) para poder auditar cada decisión.
    """
    t = _normalizar(texto)
    envio = detectar_envio(t)
    if envio:
        return "Envío", f"envio: {envio}"
    producto = detectar_producto(t)
    if producto:
        return "Producto", f"producto: {producto}"
    return "Sentimiento", "sin problema específico"


if __name__ == "__main__":
    # Pruebas rápidas de las reglas
    ejemplos = [
        "The package arrived late and damaged!!!",
        "Fast shipping, great product, love it.",
        "Arrived quickly and well packaged. The handle broke after two weeks.",
        "No leaks at all, works great.",
        "Doesn't work. Total junk.",
        "It's ok. Not what I expected but fine for the price.",
        "The box was crushed and the glass arrived shattered.",
        "Never arrived. UPS lost it.",
        "Smells awful, like chemicals.",
        "Love it! Best purchase ever.",
    ]
    for e in ejemplos:
        print(etiquetar_resena(e), "<-", e)
