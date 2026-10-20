"""Tests de `_pair_score` (nodes.py) — le coeur du scoring de similarité.

Le test le plus important du fichier est `test_two_empty_phones_do_not_match`
: c'est la régression trouvée en testant sur un vrai sandbox Salesforce
(voir README, section "Piège vécu..."), où deux contacts sans téléphone ET
sans email se retrouvaient avec un score gonflé artificiellement parce que
`"" == ""` et `fuzz.ratio("", "")` valent tous les deux "match parfait".
"""

from __future__ import annotations

from sf_dedup_poc.nodes import _digits, _pair_score

from .conftest import make_contact


def test_identical_contact_scores_100():
    a = make_contact("C1")
    b = make_contact("C2")
    score, matched = _pair_score(a, b)
    assert score == 100.0
    assert set(matched) == {"name", "email", "phone"}


def test_completely_different_contacts_score_low():
    a = make_contact("C1", first_name="Jean", last_name="Dupont", email="jean.dupont@acme.fr", phone="0612345678")
    b = make_contact(
        "C2", first_name="Marie", last_name="Lefevre", email="marie.lefevre@other.fr", phone="0698765432"
    )
    score, matched = _pair_score(a, b)
    assert score < 55.0
    assert matched == []


def test_two_empty_phones_do_not_match():
    """Régression : avant le fix, deux contacts avec un téléphone vide
    obtenaient phone_score=100 (faux positif). Doit maintenant valoir 0."""
    a = make_contact("C1", phone="")
    b = make_contact("C2", phone="")
    score, matched = _pair_score(a, b)
    assert "phone" not in matched
    # Le nom + l'email matchent toujours (mêmes valeurs par défaut), mais pas
    # le téléphone : le composite doit être strictement inférieur au cas 100%.
    assert score < 100.0


def test_two_empty_emails_do_not_match():
    a = make_contact("C1", email="")
    b = make_contact("C2", email="")
    score, matched = _pair_score(a, b)
    assert "email" not in matched
    assert score < 100.0


def test_one_empty_one_filled_phone_does_not_match():
    a = make_contact("C1", phone="0612345678")
    b = make_contact("C2", phone="")
    score, matched = _pair_score(a, b)
    assert "phone" not in matched


def test_empty_fields_on_both_sides_never_inflate_score_above_name_only():
    """Cas réel du bug observé sur le sandbox (paire "Desiree Gardner" /
    "Drew Gardner", score gonflé à 76.4 avant le fix) : avec téléphone et
    email vides des deux côtés, ces deux champs ne doivent contribuer à
    aucun signal positif — seul le nom (partiellement similaire) peut
    matcher. Sans la garde sur les champs vides, "phone" apparaissait aussi
    dans matched, ce qui faisait passer la paire dans needs_review à tort."""
    a = make_contact("C1", first_name="Desiree", last_name="Gardner", email="", phone="")
    b = make_contact("C2", first_name="Drew", last_name="Gardner", email="", phone="")
    score, matched = _pair_score(a, b)
    assert "email" not in matched
    assert "phone" not in matched
    assert score < 65.0  # sous le seuil is_duplicate de l'heuristique (judge.py)


def test_matching_phone_with_different_formatting():
    """Régression complémentaire trouvée en écrivant ces tests (voir
    `_digits` dans nodes.py) : "+33 6..." et "06..." sont le même numéro
    français, mais ne matchaient pas tant que le préfixe pays "33" n'était
    pas normalisé comme le zéro de tête."""
    a = make_contact("C1", phone="+33 6 12 34 56 78")
    b = make_contact("C2", phone="0612345678")
    assert _digits(a["phone"]) == _digits(b["phone"])
    score, matched = _pair_score(a, b)
    assert "phone" in matched


def test_digits_normalizes_country_code_and_leading_zero():
    assert _digits("+33 6 12 34 56 78") == _digits("06 12 34 56 78") == "612345678"


def test_digits_does_not_strip_country_code_from_short_numbers():
    """Le préfixe '33' n'est retiré que pour un numéro complet à 11 chiffres
    (indicatif pays + 9 chiffres) — pas question de tronquer un numéro plus
    court qui commencerait par hasard par '33'."""
    assert _digits("3312345") == "3312345"


def test_nickname_still_scores_high_on_name():
    a = make_contact("C1", first_name="Alexandre")
    b = make_contact("C2", first_name="Alex")
    score, _ = _pair_score(a, b)
    # Pas un match parfait (le nom diffère), mais le nom de famille identique
    # doit maintenir un score notable.
    assert score > 50.0
