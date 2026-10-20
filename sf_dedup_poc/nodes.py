"""
Noeuds du graphe de dédoublonnage.

Un noeud LangGraph est une fonction `(state) -> dict` : elle lit ce dont elle
a besoin dans le state courant, fait son travail, et retourne uniquement les
clés qu'elle veut mettre à jour.

Premier noeud : `block_and_match`.

Sur un vrai org Salesforce, comparer chaque Contact à tous les autres serait
O(n²) — invivable dès quelques dizaines de milliers de lignes. La technique
standard en data quality s'appelle le "blocking" : on regroupe d'abord les
enregistrements par une clé peu coûteuse à calculer et qui a de bonnes
chances de mettre les vrais doublons dans le même groupe (ici : nom de
famille normalisé + initiale du prénom). On ne fait ensuite du scoring fin
(fuzzy matching) qu'à l'intérieur de chaque groupe.
"""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from itertools import combinations

from rapidfuzz import fuzz

from langgraph.types import interrupt

from .enrichment import GENERIC_EMAIL_DOMAINS, enrich_domain
from .judge import get_chat_model, judge_pair
from .state import CandidatePair, DedupState, MergeAction, PairJudgment

# En dessous, on considère que ce n'est pas un doublon : on ne garde même
# pas la paire.
CANDIDATE_THRESHOLD = 55.0

# Au-dessus, le score déterministe suffit à trancher tout seul (pas besoin
# de solliciter le LLM). Entre les deux : zone "ambiguë", envoyée au LLM.
AUTO_CONFIDENCE_THRESHOLD = 92.0

# Confiance du jugement (LLM ou heuristique) à partir de laquelle on
# déclenche le merge automatiquement, sans repasser par un humain.
MERGE_CONFIDENCE_THRESHOLD = 0.85


def _normalize(text: str) -> str:
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    return text.strip().lower()


def _blocking_key(contact: dict) -> str:
    last = _normalize(contact["last_name"])
    first_initial = _normalize(contact["first_name"])[:1]
    return f"{last}:{first_initial}"


def _digits(phone: str) -> str:
    """Normalise un numéro de téléphone français vers une forme comparable.

    Trouvé en écrivant les tests (voir tests/test_pair_score.py) : sans ce
    fix, "+33 6 12 34 56 78" et "06 12 34 56 78" — le même numéro, deux
    formats qu'on croise réellement dans `mock_data._mutate_phone` et sur un
    vrai sandbox — ne matchaient PAS, parce qu'on ne retirait que les zéros
    de tête, jamais le préfixe pays "33". Résultat : un faux négatif (deux
    fiches de la même personne jugées non-doublons) — symétrique du bug
    "champs vides" déjà corrigé, qui lui causait des faux positifs.
    """
    digits = "".join(ch for ch in phone if ch.isdigit())
    if digits.startswith("33") and len(digits) == 11:
        # "33612345678" (indicatif pays + 9 chiffres) -> "612345678"
        digits = digits[2:]
    return digits.lstrip("0")


def _pair_score(a: dict, b: dict) -> tuple[float, list[str]]:
    """Calcule un score composite 0-100 entre deux contacts, en combinant
    plusieurs signaux, et renvoie aussi la liste des champs qui ont
    contribué le plus (utile plus tard pour expliquer la décision à un
    humain ou à un LLM).

    Piège vécu en testant sur un vrai sandbox (données de démo Salesforce,
    beaucoup de champs vides) : `fuzz.ratio("", "")` et `"" == ""` valent
    tous les deux 100/"vrai". Deux contacts qui n'ont NI l'un NI l'autre de
    téléphone (ou d'email) ne "matchent" pas pour autant sur ce champ — on
    exige explicitement que les deux valeurs soient non vides avant de
    compter un signal comme positif. Sans cette garde, deux contacts avec
    juste le même nom de famille et deux champs vides se retrouvent avec un
    score gonflé artificiellement.
    """
    name_score = fuzz.token_sort_ratio(
        f"{a['first_name']} {a['last_name']}", f"{b['first_name']} {b['last_name']}"
    )

    email_local_a = a["email"].split("@")[0] if a["email"] else ""
    email_local_b = b["email"].split("@")[0] if b["email"] else ""
    if email_local_a and email_local_b:
        email_score = fuzz.ratio(_normalize(email_local_a), _normalize(email_local_b))
    else:
        email_score = 0.0

    digits_a, digits_b = _digits(a["phone"]), _digits(b["phone"])
    phone_score = 100.0 if digits_a and digits_b and digits_a == digits_b else 0.0

    # Pondération : le nom compte le plus, mais un téléphone identique est un
    # signal fort même si le nom a beaucoup varié (surnom, faute de frappe).
    composite = 0.5 * name_score + 0.3 * email_score + 0.2 * phone_score

    matched_fields = []
    if name_score >= 80:
        matched_fields.append("name")
    if email_score >= 80:
        matched_fields.append("email")
    if phone_score == 100.0:
        matched_fields.append("phone")

    return composite, matched_fields


def block_and_match(state: DedupState) -> dict:
    contacts = state["raw_contacts"]

    blocks: dict[str, list[dict]] = defaultdict(list)
    for contact in contacts:
        blocks[_blocking_key(contact)].append(contact)

    candidate_pairs: list[CandidatePair] = []
    for block_contacts in blocks.values():
        if len(block_contacts) < 2:
            continue
        for a, b in combinations(block_contacts, 2):
            score, matched_fields = _pair_score(a, b)
            if score >= CANDIDATE_THRESHOLD:
                candidate_pairs.append(
                    CandidatePair(
                        id_a=a["id"],
                        id_b=b["id"],
                        score=round(score, 1),
                        matched_fields=matched_fields,
                    )
                )

    # On trie du plus au moins probable : plus pratique à l'affichage et,
    # plus tard, pour décider quelles paires envoyer en priorité au noeud
    # de jugement LLM / à la revue humaine.
    candidate_pairs.sort(key=lambda p: p["score"], reverse=True)

    return {"candidate_pairs": candidate_pairs}


def judge_pairs(state: DedupState) -> dict:
    """Pour chaque paire candidate, rend une décision (doublon ou non) :
    - au-dessus de AUTO_CONFIDENCE_THRESHOLD, le score suffit, pas besoin du
      LLM (économise de la latence et des coûts sur les cas évidents) ;
    - en dessous, on demande au LLM de trancher, avec retries + repli
      heuristique géré par `judge_pair` (voir judge.py) si l'appel échoue.
    """
    contacts_by_id = {c["id"]: c for c in state["raw_contacts"]}
    chat_model = get_chat_model()

    judgments: list[PairJudgment] = []
    for pair in state.get("candidate_pairs", []):
        a = contacts_by_id[pair["id_a"]]
        b = contacts_by_id[pair["id_b"]]

        if pair["score"] >= AUTO_CONFIDENCE_THRESHOLD:
            verdict = judge_pair(pair, a, b, chat_model=None)  # force le mode heuristique
        else:
            verdict = judge_pair(pair, a, b, chat_model=chat_model)

        judgments.append(
            PairJudgment(
                id_a=pair["id_a"],
                id_b=pair["id_b"],
                score=pair["score"],
                is_duplicate=verdict.is_duplicate,
                confidence=verdict.confidence,
                reason=verdict.reason,
                source=verdict.source,
            )
        )

    return {"judgments": judgments}


def split_by_confidence(state: DedupState) -> dict:
    """Répartit les jugements "doublon" en deux paniers : confiance
    suffisante pour merger tout de suite, ou pas — auquel cas on ira les
    faire valider par un humain avant de toucher à quoi que ce soit."""
    high_confidence: list[PairJudgment] = []
    needs_review: list[PairJudgment] = []

    for judgment in state.get("judgments", []):
        if not judgment["is_duplicate"]:
            continue
        if judgment["confidence"] >= MERGE_CONFIDENCE_THRESHOLD:
            high_confidence.append(judgment)
        else:
            needs_review.append(judgment)

    return {"high_confidence": high_confidence, "needs_review": needs_review}


def route_after_split(state: DedupState) -> str:
    """Fonction de routage utilisée par un edge conditionnel : elle ne
    modifie pas le state, elle renvoie juste le nom du prochain noeud. S'il
    n'y a rien d'ambigu, on saute carrément l'étape de revue humaine."""
    return "human_review" if state.get("needs_review") else "merge_duplicates"


def human_review(state: DedupState) -> dict:
    """Point d'interruption "human-in-the-loop". `interrupt(payload)` fige
    l'exécution du graphe ici et renvoie `payload` à l'appelant ; quand
    celui-ci relance le graphe avec `Command(resume=decisions)`, cette
    fonction reprend sa route exactement où elle s'était arrêtée, avec
    `decisions` comme valeur de retour de `interrupt(...)`.

    Ça suppose un checkpointer sur le graphe compilé (voir graph.py) : c'est
    lui qui permet à LangGraph de "se souvenir" où on en était pendant la
    pause, potentiellement des heures plus tard ou dans un tout autre
    process (ex: la validation humaine arrive via un bouton Slack)."""
    pending = state.get("needs_review", [])
    contacts_by_id = {c["id"]: c for c in state["raw_contacts"]}

    payload = [
        {
            "id_a": j["id_a"],
            "id_b": j["id_b"],
            "score": j["score"],
            "confidence": j["confidence"],
            "reason": j["reason"],
            "contact_a": contacts_by_id[j["id_a"]],
            "contact_b": contacts_by_id[j["id_b"]],
        }
        for j in pending
    ]
    decisions: dict[str, bool] = interrupt({"pending_reviews": payload}) or {}

    reviewed: list[PairJudgment] = []
    log_lines: list[str] = []
    for j in pending:
        key = f"{j['id_a']}:{j['id_b']}"
        approved = bool(decisions.get(key, False))
        reviewed.append({**j, "is_duplicate": approved, "source": "human"})
        verdict = "approuvé" if approved else "rejeté"
        log_lines.append(
            f"Revue humaine {j['id_a']}/{j['id_b']} "
            f"(score={j['score']}, confiance LLM={j['confidence']:.2f}) -> {verdict}"
        )

    return {"reviewed": reviewed, "log": log_lines}


def _pick_survivor(a: dict, b: dict) -> tuple[dict, dict]:
    """Choisit quel enregistrement garder lors d'un rapprochement.

    Deux règles, dans cet ordre :
    1. Entre un Contact et un Lead, on garde le Contact — c'est déjà un
       enregistrement "qualifié" côté CRM (et de toute façon Salesforce ne
       permet pas de faire un vrai merge entre les deux types d'objet, voir
       `merge_duplicates`).
    2. Entre deux enregistrements du même type, on préfère celui avec un
       email pro (domaine différent des grands webmails) plutôt qu'un
       email perso ; à égalité, l'id le plus petit, pour rester
       déterministe.
    """
    a_type = a.get("object_type", "Contact")
    b_type = b.get("object_type", "Contact")
    if a_type != b_type:
        return (a, b) if a_type == "Contact" else (b, a)

    def is_generic(contact: dict) -> bool:
        return contact["email"].split("@")[-1].lower() in GENERIC_EMAIL_DOMAINS

    a_generic, b_generic = is_generic(a), is_generic(b)
    if a_generic and not b_generic:
        return b, a
    if b_generic and not a_generic:
        return a, b
    return (a, b) if a["id"] <= b["id"] else (b, a)


def merge_duplicates(state: DedupState) -> dict:
    """Applique les rapprochements décidés (haute confiance + revus par un
    humain). Dans ce POC on ne touche évidemment à aucune donnée réelle : on
    décrit juste l'action qu'on aurait exécutée — un vrai node ferait ici un
    appel Bulk API / Apex `Database.merge` (même type d'objet uniquement),
    ou déclencherait un rapprochement Lead -> Contact sinon."""
    contacts_by_id = {c["id"]: c for c in state["raw_contacts"]}

    to_merge = list(state.get("high_confidence", [])) + [
        j for j in state.get("reviewed", []) if j["is_duplicate"]
    ]

    merges: list[MergeAction] = []
    log_lines: list[str] = []
    for j in to_merge:
        a, b = contacts_by_id[j["id_a"]], contacts_by_id[j["id_b"]]
        survivor, absorbed = _pick_survivor(a, b)
        same_type = survivor.get("object_type", "Contact") == absorbed.get("object_type", "Contact")
        kind = "merge" if same_type else "conversion_candidate"

        merges.append(
            MergeAction(
                survivor_id=survivor["id"],
                absorbed_id=absorbed["id"],
                reason=j["reason"],
                source=j["source"],
                kind=kind,
            )
        )
        if same_type:
            log_lines.append(
                f"MERGE {absorbed['id']} -> {survivor['id']} (source={j['source']}, raison: {j['reason']})"
            )
        else:
            log_lines.append(
                f"CONVERSION CANDIDATE : Lead {absorbed['id']} semble être le même contact que "
                f"{survivor['id']} (source={j['source']}) — pas de merge API direct, "
                "à traiter via un rapprochement/conversion de Lead."
            )

    return {"merges": merges, "log": log_lines}


def enrich_contacts(state: DedupState) -> dict:
    """Enrichit chaque fiche "survivante" (celles qu'aucun merge n'a
    absorbées) avec un signal de validité de domaine email (MX) et un nom
    d'entreprise deviné — voir enrichment.py pour le détail des deux
    signaux et le choix de l'API (DNS-over-HTTPS Google, gratuite et sans
    clé).

    Placé APRÈS `merge_duplicates` volontairement : on n'enrichit que les
    fiches qui vont réellement rester dans le CRM, jamais celles qui vont
    être absorbées par un merge — inutile de gaspiller un appel réseau sur
    une fiche sur le point de disparaître (le genre d'économie dont on a
    discuté en entretien technique à propos des rate limits)."""
    absorbed_ids = {m["absorbed_id"] for m in state.get("merges", [])}
    survivors = [c for c in state["raw_contacts"] if c["id"] not in absorbed_ids]

    # Mémoïsation par domaine : si plusieurs fiches survivantes partagent le
    # même domaine email, un seul appel réseau suffit pour toutes.
    cache: dict[str, bool | None] = {}

    enrichments: dict[str, dict] = {}
    for contact in survivors:
        enrichments[contact["id"]] = enrich_domain(contact["email"], cache=cache)

    return {"enrichments": enrichments}


def log_and_monitor(state: DedupState) -> dict:
    """Dernier noeud : un résumé "monitoring". Sur le SSR Stack, ce serait
    exactement le genre de payload qu'on pousserait vers un webhook Slack en
    fin de synchronisation."""
    judgments = state.get("judgments", [])
    merges = state.get("merges", [])
    enrichments = state.get("enrichments", {})
    by_source = Counter(j["source"] for j in judgments)
    invalid_domains = sum(1 for e in enrichments.values() if e["mx_valid"] is False)

    summary = [
        "--- Rapport de monitoring dédoublonnage ---",
        f"Paires évaluées : {len(judgments)}",
        f"Décisions par source : {dict(by_source)}",
        f"Merges effectués : {len(merges)}",
        f"Fiches enrichies : {len(enrichments)} (dont {invalid_domains} domaine(s) email invalide(s))",
    ]
    return {"log": summary}
