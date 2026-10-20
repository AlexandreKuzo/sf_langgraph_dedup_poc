"""Fixtures partagées : un petit helper pour fabriquer des contacts de test
sans avoir à ré-écrire les 8 champs à chaque fois. Toutes les valeurs ont un
défaut réaliste et non vide, précisément parce que le bug qu'on a corrigé
(champs vides qui matchent entre eux) est facile à recréer PAR ACCIDENT si
les tests eux-mêmes laissent des champs vides sans le vouloir.
"""

from __future__ import annotations

import pytest


def make_contact(
    id: str,
    first_name: str = "Jean",
    last_name: str = "Dupont",
    email: str = "jean.dupont@acme.fr",
    phone: str = "0612345678",
    company: str = "Acme",
    city: str = "Paris",
    object_type: str = "Contact",
    duplicate_of: str | None = None,
) -> dict:
    return {
        "id": id,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone": phone,
        "company": company,
        "city": city,
        "object_type": object_type,
        "duplicate_of": duplicate_of,
    }


@pytest.fixture
def contact_factory():
    return make_contact
