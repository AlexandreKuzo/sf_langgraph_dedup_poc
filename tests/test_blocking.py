"""Tests du blocking (`_blocking_key`) et de `block_and_match` : on vérifie
que le regroupement évite bien la comparaison O(n²) complète (deux contacts
dans des blocks différents ne sont jamais comparés, même s'ils seraient
similaires) et que le seuil CANDIDATE_THRESHOLD filtre bien les paires trop
faibles.
"""

from __future__ import annotations

from sf_dedup_poc.nodes import CANDIDATE_THRESHOLD, _blocking_key, block_and_match

from .conftest import make_contact


def test_blocking_key_normalizes_accents_and_case():
    a = make_contact("C1", first_name="Frédéric", last_name="Dupont")
    b = make_contact("C2", first_name="FREDERIC", last_name="DUPONT")
    assert _blocking_key(a) == _blocking_key(b)


def test_blocking_key_differs_on_last_name():
    a = make_contact("C1", last_name="Dupont")
    b = make_contact("C2", last_name="Martin")
    assert _blocking_key(a) != _blocking_key(b)


def test_block_and_match_only_compares_within_same_block():
    """Deux contacts avec le même prénom+email+téléphone mais un nom de
    famille différent tombent dans des blocks différents : ils ne doivent
    JAMAIS être proposés comme paire candidate, même si tout le reste
    matche — c'est la garantie que donne le blocking."""
    a = make_contact("C1", last_name="Dupont", email="x@acme.fr", phone="0611111111")
    b = make_contact("C2", last_name="Martin", email="x@acme.fr", phone="0611111111")
    state = {"raw_contacts": [a, b]}
    result = block_and_match(state)
    assert result["candidate_pairs"] == []


def test_block_and_match_finds_pair_within_block():
    a = make_contact("C1", first_name="Jean", last_name="Dupont")
    b = make_contact("C2", first_name="Jean", last_name="Dupont", email="j.dupont@acme.fr")
    state = {"raw_contacts": [a, b]}
    result = block_and_match(state)
    assert len(result["candidate_pairs"]) == 1
    pair = result["candidate_pairs"][0]
    assert {pair["id_a"], pair["id_b"]} == {"C1", "C2"}
    assert pair["score"] >= CANDIDATE_THRESHOLD


def test_block_and_match_sorts_pairs_by_descending_score():
    # Trois contacts dans le même block, avec des degrés de similarité
    # différents -> les paires candidates doivent sortir triées du plus
    # probable au moins probable.
    a = make_contact("C1", first_name="Jean", last_name="Dupont", email="jean.dupont@acme.fr", phone="0611111111")
    b = make_contact("C2", first_name="Jean", last_name="Dupont", email="jean.dupont@acme.fr", phone="0611111111")
    c = make_contact("C3", first_name="Jeanne", last_name="Dupont", email="autre@other.fr", phone="0699999999")
    state = {"raw_contacts": [a, b, c]}
    result = block_and_match(state)
    scores = [p["score"] for p in result["candidate_pairs"]]
    assert scores == sorted(scores, reverse=True)


def test_block_and_match_skips_singleton_blocks():
    """Un seul contact dans un block -> pas de paire du tout, pas d'erreur."""
    a = make_contact("C1", last_name="Unique")
    state = {"raw_contacts": [a]}
    result = block_and_match(state)
    assert result["candidate_pairs"] == []
