"""
Le "State" est le coeur de LangGraph : c'est l'objet (un dict typé, ou un
modèle pydantic) qui circule de noeud en noeud. Chaque noeud reçoit le state
courant et retourne un dict partiel avec les clés qu'il souhaite mettre à
jour ; LangGraph fusionne ce dict dans le state global avant d'appeler le
noeud suivant.

Par défaut, une clé est simplement *remplacée* par la dernière valeur
retournée par un noeud (comportement "last write wins"). Quand plusieurs
noeuds doivent au contraire *accumuler* dans une même clé (ex: un log
d'événements), on utilise un reducer via `Annotated[list[X], operator.add]`.
On introduira ça plus tard quand on ajoutera le noeud de logging/monitoring.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class CandidatePair(TypedDict):
    id_a: str
    id_b: str
    score: float
    matched_fields: list[str]


class PairJudgment(TypedDict):
    id_a: str
    id_b: str
    score: float
    is_duplicate: bool
    confidence: float
    reason: str
    source: str  # "heuristic" | "heuristic-fallback" | "llm" | "human"


class MergeAction(TypedDict):
    survivor_id: str
    absorbed_id: str
    reason: str
    source: str
    # "merge" (même type d'objet, un vrai merge Salesforce est possible) ou
    # "conversion_candidate" (Lead vs Contact : pas de merge API entre les
    # deux, il faudrait rapprocher/convertir le Lead à la place).
    kind: str


class EnrichmentRecord(TypedDict):
    domain: str | None
    # None = pas vérifié (domaine générique, pas d'email, ou échec réseau
    # après retries) — à ne jamais confondre avec False (domaine résolu mais
    # sans MX, donc probablement invalide).
    mx_valid: bool | None
    guessed_company: str | None
    # "dns" | "dns-unavailable" | "skipped-generic-domain" | "skipped-no-email"
    source: str


class DedupState(TypedDict, total=False):
    # Entrée : les contacts bruts (liste de dicts), tels qu'on les
    # récupérerait d'une requête SOQL sur Contact/Lead.
    raw_contacts: list[dict]

    # Sortie du noeud de blocking + fuzzy matching : les paires suspectées
    # d'être des doublons, avec leur score de similarité (seuil bas, on
    # inclut volontairement la zone "ambiguë" pour que le noeud suivant
    # puisse trancher).
    candidate_pairs: list[CandidatePair]

    # Sortie du noeud de jugement : une décision par paire, avec sa
    # provenance (heuristique déterministe si le score est très haut, sinon
    # LLM — ou heuristique de repli si l'appel LLM a échoué).
    judgments: list[PairJudgment]

    # Sortie de split_by_confidence : les jugements "doublon" partagés en
    # deux paniers selon la confiance du LLM/de l'heuristique.
    high_confidence: list[PairJudgment]
    needs_review: list[PairJudgment]

    # Sortie de human_review : les jugements de needs_review, réévalués par
    # un humain (source devient "human").
    reviewed: list[PairJudgment]

    # Sortie de merge_duplicates : les fusions décidées (on ne modifie pas
    # réellement de données Salesforce dans ce POC, on décrit juste l'action
    # qu'on aurait exécutée).
    merges: list[MergeAction]

    # Sortie de enrich_contacts : un EnrichmentRecord par fiche "survivante"
    # (absorbée par aucun merge), indexé par id de contact. On n'enrichit
    # que les survivants — pas la peine de gaspiller un appel réseau sur une
    # fiche qui va être fusionnée et disparaître.
    enrichments: dict[str, EnrichmentRecord]

    # Journal d'événements, alimenté par plusieurs noeuds (human_review,
    # merge_duplicates, log_and_monitor). Contrairement aux clés
    # précédentes, on veut ACCUMULER plutôt que remplacer : d'où le reducer
    # `operator.add`, qui dit à LangGraph de faire `ancienne_valeur + nouvelle_valeur`
    # (une concaténation de listes) au lieu du "dernier écrit gagne" par défaut.
    log: Annotated[list[str], operator.add]
