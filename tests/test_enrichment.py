"""Tests de enrichment.py. On ne fait jamais de vrai appel réseau ici : on
monkeypatch `enrichment._query_mx` directement (plutôt que `requests.get`)
pour éviter aussi de déclencher les vraies pauses de `tenacity` en cas
d'échec simulé — les tests doivent rester rapides et déterministes, comme
pour judge.py."""

from __future__ import annotations

import tenacity
import requests

from sf_dedup_poc.enrichment import (
    GENERIC_EMAIL_DOMAINS,
    _check_mx,
    _extract_domain,
    _guess_company_name,
    _query_mx,
    enrich_domain,
)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_extract_domain_basic():
    assert _extract_domain("jean.dupont@acme.fr") == "acme.fr"


def test_extract_domain_no_at_sign_returns_none():
    assert _extract_domain("pas-un-email") is None


def test_extract_domain_empty_returns_none():
    assert _extract_domain("") is None


def test_guess_company_name_simple_domain():
    assert _guess_company_name("acme.fr") == "Acme"


def test_guess_company_name_hyphenated_domain():
    assert _guess_company_name("acme-corp.fr") == "Acme Corp"


def test_generic_domain_is_skipped_without_any_network_call(monkeypatch):
    def _fail_if_called(domain):
        raise AssertionError("ne doit jamais être appelé pour un domaine générique")

    monkeypatch.setattr("sf_dedup_poc.enrichment._query_mx", _fail_if_called)

    domain = next(iter(GENERIC_EMAIL_DOMAINS))
    record = enrich_domain(f"jean.dupont@{domain}")
    assert record["source"] == "skipped-generic-domain"
    assert record["mx_valid"] is None
    assert record["guessed_company"] is None


def test_no_email_is_skipped():
    record = enrich_domain("")
    assert record["source"] == "skipped-no-email"
    assert record["domain"] is None


def test_valid_mx_domain(monkeypatch):
    monkeypatch.setattr(
        "sf_dedup_poc.enrichment._query_mx",
        lambda domain: _FakeResponse({"Status": 0, "Answer": [{"data": "10 mx.acme.fr."}]}),
    )
    record = enrich_domain("jean.dupont@acme.fr")
    assert record["mx_valid"] is True
    assert record["guessed_company"] == "Acme"
    assert record["source"] == "dns"


def test_domain_without_mx_records():
    def _no_mx(domain):
        return _FakeResponse({"Status": 0, "Answer": []})

    import sf_dedup_poc.enrichment as enrichment_module

    original = enrichment_module._query_mx
    enrichment_module._query_mx = _no_mx
    try:
        record = enrich_domain("jean.dupont@domaine-sans-mx.fr")
    finally:
        enrichment_module._query_mx = original
    assert record["mx_valid"] is False
    assert record["source"] == "dns"


def test_nonexistent_domain_nxdomain(monkeypatch):
    monkeypatch.setattr(
        "sf_dedup_poc.enrichment._query_mx",
        lambda domain: _FakeResponse({"Status": 3, "Answer": []}),  # NXDOMAIN
    )
    record = enrich_domain("jean.dupont@ce-domaine-nexiste-pas-xyz.fr")
    assert record["mx_valid"] is False


def test_network_failure_after_retries_returns_unverified_not_invalid(monkeypatch):
    """Un souci réseau ne doit JAMAIS être interprété comme "domaine
    invalide" (False) — seulement comme "pas vérifié" (None), pour ne pas
    fausser un vrai jugement de data quality avec un problème transitoire
    côté infra."""

    def _always_fails(domain):
        raise requests.exceptions.ConnectionError("simulated network outage")

    monkeypatch.setattr("sf_dedup_poc.enrichment._query_mx", _always_fails)

    record = enrich_domain("jean.dupont@acme.fr")
    assert record["mx_valid"] is None
    assert record["source"] == "dns-unavailable"
    # Le nom d'entreprise deviné (heuristique locale, sans réseau) reste
    # disponible même si la vérification MX a échoué.
    assert record["guessed_company"] == "Acme"


def test_exhausted_retries_never_leak_a_bare_retry_error(monkeypatch):
    """Régression sur `_query_mx` lui-même (pas seulement `_check_mx`) :
    sans `reraise=True` sur le décorateur @retry, tenacity enveloppe
    l'exception d'origine dans un tenacity.RetryError une fois les 3
    tentatives épuisées — un type que `_check_mx` (qui n'attrape que
    RequestException) ne rattrape pas, et qui remontait donc jusqu'à faire
    planter le noeud. On désactive juste l'attente entre tentatives pour ne
    pas ralentir la suite, sans modifier le comportement testé."""
    monkeypatch.setattr(_query_mx.retry, "wait", tenacity.wait_none())
    monkeypatch.setattr(
        "requests.get",
        lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.ConnectionError("simulated outage")),
    )

    # `_query_mx` doit laisser passer l'exception RÉSEAU d'origine (grâce à
    # reraise=True), jamais un tenacity.RetryError.
    try:
        _query_mx("acme.fr")
        assert False, "aurait dû lever une exception"
    except tenacity.RetryError:
        assert False, "reraise=True manquant : RetryError ne doit jamais fuiter"
    except requests.exceptions.RequestException:
        pass  # comportement attendu

    # Et `_check_mx`, qui attrape RequestException, doit donc dégrader
    # proprement plutôt que de laisser fuiter quoi que ce soit.
    assert _check_mx("acme.fr") is None


def test_cache_avoids_second_network_call_for_same_domain():
    call_count = {"n": 0}

    def _counting_query(domain):
        call_count["n"] += 1
        return _FakeResponse({"Status": 0, "Answer": [{"data": "10 mx.acme.fr."}]})

    import sf_dedup_poc.enrichment as enrichment_module

    original = enrichment_module._query_mx
    enrichment_module._query_mx = _counting_query
    try:
        cache: dict = {}
        enrich_domain("jean.dupont@acme.fr", cache=cache)
        enrich_domain("marie.martin@acme.fr", cache=cache)  # même domaine
    finally:
        enrichment_module._query_mx = original

    assert call_count["n"] == 1
