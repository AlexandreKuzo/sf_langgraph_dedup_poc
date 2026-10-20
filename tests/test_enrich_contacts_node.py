"""Tests du noeud `enrich_contacts` (nodes.py) : la partie qui lui est
propre est le filtrage "uniquement les survivants" — la logique de scoring
MX elle-même est déjà couverte par test_enrichment.py. On monkeypatch le
réseau au même niveau (`_query_mx`) pour rester déterministe et rapide."""

from __future__ import annotations

from sf_dedup_poc.nodes import enrich_contacts

from .conftest import make_contact


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _patch_query_mx(monkeypatch, response_payload=None, counter=None):
    def _fake(domain):
        if counter is not None:
            counter["n"] = counter.get("n", 0) + 1
        return _FakeResponse(response_payload or {"Status": 0, "Answer": [{"data": "10 mx.acme.fr."}]})

    monkeypatch.setattr("sf_dedup_poc.enrichment._query_mx", _fake)


def test_absorbed_contacts_are_not_enriched(monkeypatch):
    counter = {}
    _patch_query_mx(monkeypatch, counter=counter)

    survivor = make_contact("C1", email="jean.dupont@acme.fr")
    absorbed = make_contact("C2", email="jean.dupont@othercorp.fr")
    state = {
        "raw_contacts": [survivor, absorbed],
        "merges": [
            {"survivor_id": "C1", "absorbed_id": "C2", "reason": "x", "source": "heuristic", "kind": "merge"}
        ],
    }
    result = enrich_contacts(state)

    assert set(result["enrichments"].keys()) == {"C1"}
    # Un seul appel réseau : uniquement pour le survivant, pas pour l'absorbé.
    assert counter["n"] == 1


def test_no_merges_enriches_everyone(monkeypatch):
    _patch_query_mx(monkeypatch)
    a = make_contact("C1", email="jean.dupont@acme.fr")
    b = make_contact("C2", email="marie.martin@othercorp.fr")
    state = {"raw_contacts": [a, b], "merges": []}
    result = enrich_contacts(state)
    assert set(result["enrichments"].keys()) == {"C1", "C2"}


def test_shared_domain_across_survivors_hits_network_once(monkeypatch):
    counter = {}
    _patch_query_mx(monkeypatch, counter=counter)
    a = make_contact("C1", email="jean.dupont@acme.fr")
    b = make_contact("C2", email="marie.martin@acme.fr")
    state = {"raw_contacts": [a, b], "merges": []}
    enrich_contacts(state)
    assert counter["n"] == 1
