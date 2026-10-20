"""Tests de mock_data.py : surtout le déterminisme (seed fixe), qui est ce
qui permet de comparer les résultats du graphe à une "vérité terrain"
connue dans run.py (`has_ground_truth=True` seulement pour la source mock)."""

from __future__ import annotations

from sf_dedup_poc.mock_data import generate_contacts


def test_same_seed_produces_identical_output():
    a = generate_contacts(n_base=20, dup_rate=0.5, seed=7)
    b = generate_contacts(n_base=20, dup_rate=0.5, seed=7)
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]


def test_different_seed_produces_different_output():
    a = generate_contacts(n_base=20, dup_rate=0.5, seed=1)
    b = generate_contacts(n_base=20, dup_rate=0.5, seed=2)
    assert [c.to_dict() for c in a] != [c.to_dict() for c in b]


def test_all_ids_are_unique():
    contacts = generate_contacts(n_base=40, dup_rate=0.5, seed=42)
    ids = [c.id for c in contacts]
    assert len(ids) == len(set(ids))


def test_duplicate_of_always_points_to_an_existing_id():
    contacts = generate_contacts(n_base=40, dup_rate=0.5, seed=42)
    ids = {c.id for c in contacts}
    for c in contacts:
        if c.duplicate_of is not None:
            assert c.duplicate_of in ids


def test_generates_at_least_some_duplicates_with_default_rate():
    contacts = generate_contacts(n_base=40, dup_rate=0.5, seed=42)
    n_dups = sum(1 for c in contacts if c.duplicate_of is not None)
    assert n_dups > 0


def test_zero_dup_rate_generates_no_duplicates():
    contacts = generate_contacts(n_base=20, dup_rate=0.0, seed=42)
    assert all(c.duplicate_of is None for c in contacts)
    assert len(contacts) == 20
