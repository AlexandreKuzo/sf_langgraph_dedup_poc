"""
Enrichissement de fiches à partir du domaine email.

Couvre le troisième volet "data quality" de l'offre Skello (le premier étant
le dédoublonnage, le deuxième la fiabilité des appels API) : une fois qu'on
sait quelle fiche on garde (`merge_duplicates`), on l'enrichit avec deux
signaux dérivés du domaine email :

1. **Validité du domaine (MX)** : un vrai appel réseau, à l'API publique
   DNS-over-HTTPS de Google (`https://dns.google/resolve`) — gratuite, sans
   clé, stable et documentée (https://developers.google.com/speed/public-dns/docs/doh).
   On vérifie que le domaine a des enregistrements MX, c'est-à-dire qu'il
   peut recevoir des emails. C'est un signal de data quality concret : une
   fiche avec un domaine email qui n'existe pas (faute de frappe, domaine
   expiré...) est aussi cassée qu'un doublon non détecté.
2. **Nom d'entreprise deviné** : une heuristique locale, sans réseau, à
   partir du nom de domaine (ex. `acme-corp.fr` -> "Acme Corp"). Un vrai
   provider d'enrichissement firmographique (Clearbit, Apollo...) ferait
   beaucoup mieux, mais suivrait exactement le même point d'extension que
   `judge.get_chat_model()` : une fonction qui renvoie un client si une clé
   API est configurée, `None` sinon.

Même philosophie de robustesse que `judge.py` et `salesforce_source.py` :
- on ne retente QUE les erreurs réseau/transitoires, jamais une réponse DNS
  qui dit légitimement "ce domaine n'a pas de MX" (ce n'est pas un échec,
  c'est une réponse) ;
- si l'appel échoue malgré les retries, on ne fait JAMAIS planter le
  graphe : on retombe sur `mx_valid=None` ("non vérifié") plutôt que de
  bloquer tout le pipeline pour un souci réseau sur un enrichissement
  secondaire.
- on ne fait qu'un seul appel réseau par domaine, même si plusieurs fiches
  survivantes partagent le même domaine (`enrich_contacts` mémoïse) — le
  genre d'économie d'appels dont il a été question en entretien technique
  (volumétrie, rate limits).
"""

from __future__ import annotations

import re

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .state import EnrichmentRecord

DNS_ENDPOINT = "https://dns.google/resolve"

# Domaines de grand public : un match dessus ne dit rien sur l'entreprise du
# contact, donc pas la peine de "deviner" un nom d'entreprise ni de gaspiller
# un appel réseau dessus. Même liste que `_pick_survivor` (nodes.py), qui
# l'importe d'ici pour ne pas la dupliquer.
GENERIC_EMAIL_DOMAINS = {"gmail.com", "yahoo.fr", "hotmail.com", "outlook.com", "orange.fr"}

_RETRYABLE = retry_if_exception_type(requests.exceptions.RequestException)


def _extract_domain(email: str) -> str | None:
    if not email or "@" not in email:
        return None
    return email.rsplit("@", 1)[-1].strip().lower() or None


def _guess_company_name(domain: str) -> str:
    """Heuristique locale, sans réseau : 'acme-corp.fr' -> 'Acme Corp'.
    Volontairement simple (mode "démo") — un vrai provider d'enrichissement
    ferait un lookup firmographique réel."""
    label = domain.split(".")[0]
    words = re.split(r"[-_.]+", label)
    return " ".join(w.capitalize() for w in words if w)


@retry(
    retry=_RETRYABLE,
    # Politique de retry volontairement plus légère que sur
    # salesforce_source.py/judge.py : contrairement à une auth Salesforce ou
    # un jugement LLM (au coeur du pipeline), l'enrichissement MX est une
    # donnée secondaire — si le réseau bloque durablement dns.google (vécu
    # en testant ce POC depuis un environnement dont le proxy sortant refuse
    # ce domaine), on préfère échouer vite sur CHAQUE domaine plutôt que de
    # ralentir tout le run de plusieurs minutes sur une volumétrie de
    # contacts avec autant de domaines distincts.
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
    # Sans reraise=True, tenacity enveloppe l'exception d'origine dans un
    # tenacity.RetryError une fois les tentatives épuisées — un type que
    # `_check_mx` (qui n'attrape que RequestException) ne rattrape pas.
    # Trouvé en testant ce noeud contre un vrai réseau restreint (proxy
    # sortant qui bloque dns.google) : sans ce paramètre, un souci réseau
    # persistant plantait avec une trace brute au lieu de dégrader
    # proprement vers mx_valid=None. Même correctif appliqué à
    # salesforce_source.py, qui avait le même trou.
    reraise=True,
)
def _query_mx(domain: str) -> requests.Response:
    return requests.get(DNS_ENDPOINT, params={"name": domain, "type": "MX"}, timeout=5)


def _check_mx(domain: str) -> bool | None:
    """Renvoie True (MX trouvé), False (domaine résolu mais sans MX /
    inexistant), ou None si la vérification elle-même a échoué (réseau
    indisponible après retries) — un None ne doit jamais être interprété
    comme "domaine invalide", juste "pas pu vérifier"."""
    try:
        response = _query_mx(domain)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    payload = response.json()
    # Status 0 = NOERROR (RFC 2136) ; Answer non vide = au moins un
    # enregistrement MX trouvé.
    return payload.get("Status") == 0 and bool(payload.get("Answer"))


def enrich_domain(email: str, *, cache: dict[str, bool | None] | None = None) -> EnrichmentRecord:
    """Point d'entrée utilisé par le noeud `enrich_contacts`. `cache` est un
    dict partagé entre plusieurs appels (fourni par le noeud) pour éviter de
    requêter deux fois le même domaine dans une même exécution."""
    domain = _extract_domain(email)
    if domain is None:
        return EnrichmentRecord(domain=None, mx_valid=None, guessed_company=None, source="skipped-no-email")

    if domain in GENERIC_EMAIL_DOMAINS:
        return EnrichmentRecord(domain=domain, mx_valid=None, guessed_company=None, source="skipped-generic-domain")

    if cache is not None and domain in cache:
        mx_valid = cache[domain]
    else:
        mx_valid = _check_mx(domain)
        if cache is not None:
            cache[domain] = mx_valid

    if mx_valid is None:
        return EnrichmentRecord(
            domain=domain, mx_valid=None, guessed_company=_guess_company_name(domain), source="dns-unavailable"
        )
    return EnrichmentRecord(
        domain=domain, mx_valid=mx_valid, guessed_company=_guess_company_name(domain), source="dns"
    )
