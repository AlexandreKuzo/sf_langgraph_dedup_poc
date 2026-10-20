"""Tests de `_pick_survivor` et `merge_duplicates` : quel enregistrement on
garde, et si l'action est un MERGE (même type d'objet) ou un CONVERSION
CANDIDATE (Lead vs Contact, pas de merge API direct chez Salesforce)."""

from __future__ import annotations

from sf_dedup_poc.nodes import _pick_survivor, merge_duplicates

from .conftest import make_contact


def test_contact_beats_lead_regardless_of_order():
    contact = make_contact("C1", object_type="Contact")
    lead = make_contact("L1", object_type="Lead")
    survivor, absorbed = _pick_survivor(contact, lead)
    assert survivor["id"] == "C1"
    assert absorbed["id"] == "L1"

    # Ordre inversé -> même résultat.
    survivor, absorbed = _pick_survivor(lead, contact)
    assert survivor["id"] == "C1"
    assert absorbed["id"] == "L1"


def test_professional_email_beats_generic_email_same_type():
    pro = make_contact("C1", email="jean.dupont@acme.fr")
    generic = make_contact("C2", email="jean.dupont@gmail.com")
    survivor, absorbed = _pick_survivor(pro, generic)
    assert survivor["id"] == "C1"
    assert absorbed["id"] == "C2"


def test_tie_break_is_deterministic_by_id():
    a = make_contact("C1", email="a@gmail.com")
    b = make_contact("C2", email="b@gmail.com")
    survivor, _ = _pick_survivor(a, b)
    assert survivor["id"] == "C1"  # id le plus petit, à égalité de tout le reste
    survivor, _ = _pick_survivor(b, a)
    assert survivor["id"] == "C1"  # ordre des arguments ne doit pas changer le résultat


def test_merge_duplicates_produces_merge_kind_for_same_object_type():
    a = make_contact("C1", object_type="Contact")
    b = make_contact("C2", object_type="Contact")
    state = {
        "raw_contacts": [a, b],
        "high_confidence": [
            {"id_a": "C1", "id_b": "C2", "score": 95.0, "is_duplicate": True,
             "confidence": 0.95, "reason": "score élevé", "source": "heuristic"}
        ],
        "reviewed": [],
    }
    result = merge_duplicates(state)
    assert len(result["merges"]) == 1
    assert result["merges"][0]["kind"] == "merge"


def test_merge_duplicates_produces_conversion_candidate_for_lead_vs_contact():
    contact = make_contact("C1", object_type="Contact")
    lead = make_contact("L1", object_type="Lead")
    state = {
        "raw_contacts": [contact, lead],
        "high_confidence": [
            {"id_a": "C1", "id_b": "L1", "score": 95.0, "is_duplicate": True,
             "confidence": 0.95, "reason": "score élevé", "source": "heuristic"}
        ],
        "reviewed": [],
    }
    result = merge_duplicates(state)
    assert result["merges"][0]["kind"] == "conversion_candidate"
    assert result["merges"][0]["survivor_id"] == "C1"
    assert result["merges"][0]["absorbed_id"] == "L1"


def test_merge_duplicates_only_merges_reviewed_pairs_approved_by_human():
    a = make_contact("C1")
    b = make_contact("C2")
    state = {
        "raw_contacts": [a, b],
        "high_confidence": [],
        "reviewed": [
            {"id_a": "C1", "id_b": "C2", "score": 60.0, "is_duplicate": False,
             "confidence": 0.5, "reason": "rejeté par l'humain", "source": "human"}
        ],
    }
    result = merge_duplicates(state)
    assert result["merges"] == []
