"""Tests de judge.py : décision heuristique (mode démo, sans clé API) et
repli automatique quand l'appel LLM échoue. On ne teste jamais un vrai appel
réseau ici — chat_model=None couvre le mode démo, et un faux chat_model qui
lève une exception couvre le chemin de repli, exactement comme le node
judge_pairs les utilise (score >= AUTO_CONFIDENCE_THRESHOLD -> chat_model=None
forcé, sinon un vrai modèle si ANTHROPIC_API_KEY est configurée)."""

from __future__ import annotations

import pytest

from sf_dedup_poc.judge import DuplicateJudgment, _heuristic_judgment, judge_pair

from .conftest import make_contact


def _pair(score: float) -> dict:
    return {"id_a": "C1", "id_b": "C2", "score": score, "matched_fields": ["name"]}


def test_heuristic_judgment_above_threshold_is_duplicate():
    verdict = _heuristic_judgment(_pair(80.0), source="heuristic")
    assert verdict.is_duplicate is True
    assert verdict.confidence == 0.8
    assert verdict.source == "heuristic"


def test_heuristic_judgment_below_threshold_is_not_duplicate():
    verdict = _heuristic_judgment(_pair(40.0), source="heuristic")
    assert verdict.is_duplicate is False


def test_heuristic_judgment_boundary_is_65():
    assert _heuristic_judgment(_pair(65.0), source="heuristic").is_duplicate is True
    assert _heuristic_judgment(_pair(64.9), source="heuristic").is_duplicate is False


def test_judge_pair_without_chat_model_uses_heuristic():
    """chat_model=None (pas de clé API, ou score déjà tranché par
    AUTO_CONFIDENCE_THRESHOLD dans judge_pairs) -> jamais d'appel réseau."""
    a, b = make_contact("C1"), make_contact("C2")
    verdict = judge_pair(_pair(90.0), a, b, chat_model=None)
    assert verdict.source == "heuristic"
    assert "mode démo" in verdict.reason


def test_judge_pair_falls_back_to_heuristic_when_llm_call_fails():
    """Simule un LLM qui échoue systématiquement (rate limit, timeout...) :
    le graphe ne doit jamais planter, `judge_pair` doit retomber sur
    l'heuristique avec source='heuristic-fallback'."""

    class _AlwaysFailingStructuredOutput:
        def invoke(self, prompt):
            raise RuntimeError("simulated 429 rate limit")

    class _AlwaysFailingChatModel:
        def with_structured_output(self, schema):
            return _AlwaysFailingStructuredOutput()

    a, b = make_contact("C1"), make_contact("C2")
    verdict = judge_pair(_pair(75.0), a, b, chat_model=_AlwaysFailingChatModel())
    assert verdict.source == "heuristic-fallback"
    assert "l'appel LLM a échoué" in verdict.reason


def test_judge_pair_uses_llm_result_when_call_succeeds():
    """Simule un LLM qui répond correctement du premier coup."""

    class _WorkingStructuredOutput:
        def invoke(self, prompt):
            return DuplicateJudgment(is_duplicate=True, confidence=0.99, reason="même personne, faute de frappe")

    class _WorkingChatModel:
        def with_structured_output(self, schema):
            return _WorkingStructuredOutput()

    a, b = make_contact("C1"), make_contact("C2")
    verdict = judge_pair(_pair(75.0), a, b, chat_model=_WorkingChatModel())
    assert verdict.source == "llm"
    assert verdict.is_duplicate is True
    assert verdict.confidence == 0.99


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_duplicate_judgment_rejects_out_of_range_confidence(confidence):
    with pytest.raises(ValueError):
        DuplicateJudgment(is_duplicate=True, confidence=confidence, reason="x")
