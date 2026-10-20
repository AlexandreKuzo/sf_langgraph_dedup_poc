"""Tests bout-en-bout du graphe compilé (`build_graph`) : pas seulement les
fonctions individuelles, mais le vrai câblage LangGraph — routage
conditionnel, et surtout le cycle interrupt()/Command(resume=...) du
human-in-the-loop, qui est la partie la plus facile à casser silencieusement
si on retouche au graphe (un edge oublié, un mauvais nom de noeud...).

`monkeypatch.delenv("ANTHROPIC_API_KEY", ...)` force le mode heuristique/démo
dans judge.py, pour que ces tests restent déterministes et ne fassent jamais
d'appel réseau, même si une vraie clé est configurée dans l'environnement de
qui les lance.
"""

from __future__ import annotations

import pytest
from langgraph.types import Command

from sf_dedup_poc.graph import build_graph

from .conftest import make_contact


class _FakeMxResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _no_real_dns_calls(monkeypatch):
    """enrich_contacts fait un vrai appel réseau (DNS-over-HTTPS) par
    défaut : on le monkeypatch ici pour que ces tests de bout en bout
    restent déterministes et rapides, indépendamment de la connectivité de
    la machine qui les lance."""
    monkeypatch.setattr(
        "sf_dedup_poc.enrichment._query_mx",
        lambda domain: _FakeMxResponse({"Status": 0, "Answer": [{"data": "10 mx.example.fr."}]}),
    )


def test_high_confidence_pair_merges_without_human_review():
    """Deux contacts identiques -> score 100 -> AUTO_CONFIDENCE_THRESHOLD
    dépassé -> heuristique directe, confiance 1.0 -> merge immédiat, sans
    jamais passer par human_review (pas de __interrupt__ dans le résultat)."""
    a = make_contact("C1")
    b = make_contact("C2")
    app = build_graph()
    config = {"configurable": {"thread_id": "test-high-confidence"}}

    result = app.invoke({"raw_contacts": [a, b]}, config=config)

    assert "__interrupt__" not in result
    assert len(result["judgments"]) == 1
    assert len(result["merges"]) == 1
    assert result["merges"][0]["survivor_id"] == "C1"
    assert result["merges"][0]["absorbed_id"] == "C2"
    # enrich_contacts tourne après merge_duplicates : seul le survivant (C1)
    # doit être enrichi, jamais C2 qui a été absorbé.
    assert set(result["enrichments"].keys()) == {"C1"}
    assert result["enrichments"]["C1"]["mx_valid"] is True


def test_ambiguous_pair_pauses_for_human_review_and_resumes_on_approval():
    """Paire dans la zone grise (score ~71, confiance heuristique ~0.71 <
    MERGE_CONFIDENCE_THRESHOLD) : le graphe doit s'arrêter sur human_review,
    puis reprendre exactement là où il s'était arrêté une fois
    Command(resume=...) envoyé, merger si on approuve."""
    a = make_contact(
        "C1", first_name="Julien", last_name="Martin",
        email="julien.martin@acme.fr", phone="0611111111",
    )
    b = make_contact(
        "C2", first_name="Julie", last_name="Martin",
        email="j.martin@other.fr", phone="0699999999",
    )
    app = build_graph()
    config = {"configurable": {"thread_id": "test-ambiguous-approve"}}

    result = app.invoke({"raw_contacts": [a, b]}, config=config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert len(payload["pending_reviews"]) == 1
    pending = payload["pending_reviews"][0]
    assert {pending["id_a"], pending["id_b"]} == {"C1", "C2"}

    key = f"{pending['id_a']}:{pending['id_b']}"
    result = app.invoke(Command(resume={key: True}), config=config)

    assert "__interrupt__" not in result
    assert len(result["merges"]) == 1
    assert result["merges"][0]["source"] == "human"


def test_ambiguous_pair_rejected_by_human_is_not_merged():
    a = make_contact(
        "C1", first_name="Julien", last_name="Martin",
        email="julien.martin@acme.fr", phone="0611111111",
    )
    b = make_contact(
        "C2", first_name="Julie", last_name="Martin",
        email="j.martin@other.fr", phone="0699999999",
    )
    app = build_graph()
    config = {"configurable": {"thread_id": "test-ambiguous-reject"}}

    result = app.invoke({"raw_contacts": [a, b]}, config=config)
    payload = result["__interrupt__"][0].value
    pending = payload["pending_reviews"][0]
    key = f"{pending['id_a']}:{pending['id_b']}"

    result = app.invoke(Command(resume={key: False}), config=config)

    assert result["merges"] == []


def test_no_candidate_pairs_skips_straight_to_end():
    """Deux contacts sans aucun rapport (blocks différents) -> aucune paire
    candidate -> le graphe doit terminer sans erreur, avec des listes vides
    partout plutôt que planter sur une clé manquante."""
    a = make_contact("C1", last_name="Dupont")
    b = make_contact("C2", last_name="Completement-Different")
    app = build_graph()
    config = {"configurable": {"thread_id": "test-no-pairs"}}

    result = app.invoke({"raw_contacts": [a, b]}, config=config)

    assert "__interrupt__" not in result
    assert result["judgments"] == []
    assert result["merges"] == []
