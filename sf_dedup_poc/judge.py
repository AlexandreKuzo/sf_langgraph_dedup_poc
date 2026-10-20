"""
Jugement des paires "ambiguës" : quand le score déterministe (blocking +
fuzzy matching) ne suffit pas à trancher tout seul, on demande à un LLM de
statuer, avec une justification structurée.

Deux points sont volontairement soignés ici, parce qu'ils correspondent
directement à ce qu'on lit dans l'offre Skello ("prendre en charge la
gestion des erreurs, des rate limits et des retries pour que le code
appelant reste simple") :

1. `judge_pair()` est le seul point de contact du node avec cette
   mécanique : le node ne sait rien des retries, il appelle juste la
   fonction et reçoit un jugement.
2. Si l'appel LLM échoue de façon persistante (rate limit, timeout, panne
   du provider...), on ne fait PAS planter tout le graphe : on retombe sur
   une décision heuristique basée sur le score, avec une raison qui dit
   explicitement qu'on est en mode dégradé. Un vrai pipeline de
   dédoublonnage ne doit jamais bloquer une synchronisation entière parce
   qu'un appel LLM a échoué.

Sans clé API configurée (`ANTHROPIC_API_KEY`), tout le POC tourne quand même
en "mode démo" : `get_chat_model()` renvoie `None` et on utilise uniquement
la décision heuristique. C'est ce qui permet de faire tourner et de
partager ce POC sans dépendance externe.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class DuplicateJudgment(BaseModel):
    is_duplicate: bool = Field(description="True si les deux contacts sont la même personne")
    confidence: float = Field(ge=0.0, le=1.0, description="Confiance dans la décision, entre 0 et 1")
    reason: str = Field(description="Justification courte, en une phrase")
    source: str = Field(default="llm", description="'llm', 'heuristic' ou 'heuristic-fallback'")


_JUDGE_PROMPT = """Tu es un expert en qualité de données Salesforce. On te donne deux \
fiches contact qui ont été rapprochées par un algorithme de similarité \
(score {score}/100, champs qui matchent déjà : {matched_fields}). Détermine \
s'il s'agit de la même personne (doublon) ou de deux personnes différentes.

Contact A: {a}
Contact B: {b}

Attention aux variations plausibles pour une même personne (surnom, casse,
email perso vs pro, faute de frappe, numéro reformaté) sans les confondre
avec deux personnes réellement différentes qui partagent juste un nom
courant ou une entreprise."""


def get_chat_model():
    """Renvoie un chat model si une clé API est configurée, sinon None (mode
    heuristique/démo, pour pouvoir faire tourner tout le POC sans
    dépendance externe). Pas de branche OpenAI ici pour rester simple, mais
    l'endroit où l'ajouter est évident."""
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    return None


def _heuristic_judgment(pair: dict, source: str, reason_suffix: str = "") -> DuplicateJudgment:
    score = pair["score"]
    return DuplicateJudgment(
        is_duplicate=score >= 65,
        confidence=round(score / 100, 2),
        reason=f"Décision heuristique basée sur le score de similarité ({score}/100)"
        + reason_suffix,
        source=source,
    )


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _call_llm(chat_model, a: dict, b: dict, pair: dict) -> DuplicateJudgment:
    structured = chat_model.with_structured_output(DuplicateJudgment)
    prompt = _JUDGE_PROMPT.format(
        score=pair["score"], matched_fields=pair["matched_fields"], a=a, b=b
    )
    result = structured.invoke(prompt)
    result.source = "llm"
    return result


def judge_pair(pair: dict, a: dict, b: dict, chat_model) -> DuplicateJudgment:
    if chat_model is None:
        return _heuristic_judgment(pair, "heuristic", " (mode démo, pas de clé API configurée)")
    try:
        return _call_llm(chat_model, a, b, pair)
    except Exception as exc:  # noqa: BLE001 - fallback volontairement large
        return _heuristic_judgment(
            pair, "heuristic-fallback", f" (l'appel LLM a échoué après retries : {exc})"
        )
