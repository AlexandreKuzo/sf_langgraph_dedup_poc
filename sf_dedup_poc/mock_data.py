"""
Génère un jeu de Contacts Salesforce factices, avec des doublons volontaires.

Objectif pédagogique : avoir un jeu de données reproductible (seed fixe) sur
lequel on peut développer et tester la logique de dédoublonnage avant de la
brancher sur un vrai sandbox Salesforce.

Chaque "vrai" contact peut générer 0, 1 ou 2 doublons construits par petites
mutations réalistes :
- variation de casse / accents dans le nom
- email avec un domaine différent, ou en tout minuscule
- téléphone reformaté (espaces, points, +33 vs 0)
- surnom à la place du prénom complet
"""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass, field, asdict

from faker import Faker

fake = Faker("fr_FR")

NICKNAMES = {
    "Alexandre": "Alex",
    "Nicolas": "Nico",
    "Frédéric": "Fred",
    "Stéphanie": "Steph",
    "Guillaume": "Guigui",
    "Mathieu": "Matt",
    "Isabelle": "Isa",
    "Vincent": "Vince",
}


@dataclass
class Contact:
    id: str
    first_name: str
    last_name: str
    email: str
    phone: str
    company: str
    city: str
    # id du "vrai" contact dont celui-ci est un doublon généré (None si original)
    duplicate_of: str | None = field(default=None, repr=False)
    # Toutes les fiches mock sont des "Contact" ; la vraie source Salesforce
    # (salesforce_source.py) peut aussi renvoyer des "Lead", d'où ce champ
    # commun aux deux sources.
    object_type: str = "Contact"

    def to_dict(self) -> dict:
        return asdict(self)


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def _mutate_name(first_name: str) -> str:
    choice = random.random()
    if choice < 0.3 and first_name in NICKNAMES:
        return NICKNAMES[first_name]
    if choice < 0.55:
        return _strip_accents(first_name)
    if choice < 0.7:
        return first_name.upper()
    return first_name


def _mutate_email(first_name: str, last_name: str, original_email: str) -> str:
    choice = random.random()
    domain = original_email.split("@")[-1]
    if choice < 0.4:
        # Domaine perso à la place du domaine pro (cas classique : le
        # commercial a saisi l'email perso par erreur)
        return f"{_strip_accents(first_name).lower()}.{_strip_accents(last_name).lower()}@gmail.com"
    if choice < 0.7:
        return original_email.upper()
    # petit typo dans le nom de domaine
    return original_email.replace(".fr", ".frr", 1) if ".fr" in original_email else original_email


def _mutate_phone(phone: str) -> str:
    digits = "".join(ch for ch in phone if ch.isdigit())
    if digits.startswith("0"):
        digits = digits[1:]
    choice = random.random()
    if choice < 0.4:
        return "+33 " + " ".join([digits[i : i + 2] for i in range(0, len(digits), 2)])
    if choice < 0.7:
        return "0" + digits
    return "0" + ".".join([digits[i : i + 2] for i in range(0, len(digits), 2)])


def generate_contacts(n_base: int = 40, dup_rate: float = 0.5, seed: int = 42) -> list[Contact]:
    """Génère n_base contacts "originaux" puis ajoute des doublons pour une
    fraction dup_rate d'entre eux (0, 1 ou 2 doublons chacun)."""
    random.seed(seed)
    Faker.seed(seed)

    contacts: list[Contact] = []
    for i in range(n_base):
        first_name = fake.first_name()
        last_name = fake.last_name()
        company = fake.company()
        contact = Contact(
            id=f"C{i:04d}",
            first_name=first_name,
            last_name=last_name,
            email=f"{_strip_accents(first_name).lower()}.{_strip_accents(last_name).lower()}@{company.lower().split()[0].strip(',.')}.fr",
            phone=fake.phone_number(),
            company=company,
            city=fake.city(),
        )
        contacts.append(contact)

        if random.random() < dup_rate:
            n_dups = random.choice([1, 1, 2])
            for d in range(n_dups):
                dup = Contact(
                    id=f"C{i:04d}-D{d}",
                    first_name=_mutate_name(first_name),
                    last_name=last_name,
                    email=_mutate_email(first_name, last_name, contact.email),
                    phone=_mutate_phone(contact.phone),
                    company=company,
                    city=contact.city,
                    duplicate_of=contact.id,
                )
                contacts.append(dup)

    random.shuffle(contacts)
    return contacts


if __name__ == "__main__":
    for c in generate_contacts(n_base=10, seed=1):
        print(c)
