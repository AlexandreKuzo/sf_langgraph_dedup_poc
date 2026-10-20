"""
Source de données "réelle" : authentification OAuth contre un sandbox
Salesforce, puis récupération de Contacts + Leads via l'API REST (SOQL),
remis à plat dans exactement la même forme que
`mock_data.generate_contacts()` — c'est ce découplage qui permet de
brancher le même graphe LangGraph sans y toucher.

Deux flows OAuth supportés via `SF_AUTH_FLOW` (voir `.env.example`) :

- `client_credentials` (par défaut) : Consumer Key + Consumer Secret d'un
  External Client App, avec un "Run As User" configuré côté Salesforce
  (Policies -> Client Credentials Flow). Pas de certificat à gérer — le
  plus simple des deux à mettre en place.
  Point important : le endpoint doit être le **My Domain** de l'org (ex.
  `https://mon-org--sandbox.sandbox.my.salesforce.com`), PAS
  `login.salesforce.com`/`test.salesforce.com` : sans étape de login
  utilisateur, Salesforce a besoin de l'URL spécifique à l'org pour savoir
  à quel External Client App s'adresser.

- `jwt` : JWT Bearer Flow — une assertion signée par une clé privée plutôt
  qu'un secret partagé. Utile si la politique de l'org l'impose à la
  place (voir le README pour la procédure de certificat).
"""

from __future__ import annotations

import os
import time
from urllib.parse import quote

import jwt
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

API_VERSION = "v61.0"

# On ne retente que les erreurs réseau/transitoires (timeout, connexion
# coupée, 5xx via raise_for_status) — jamais nos propres SalesforceAuthError
# (mauvais identifiants, flow mal configuré, token expiré...) : ça ne se
# répare pas en retentant, autant échouer vite avec un message clair.
_RETRYABLE = retry_if_exception_type(requests.exceptions.RequestException)


class SalesforceAuthError(Exception):
    pass


def _require_env(*names: str) -> dict:
    config = {name: os.getenv(name) for name in names}
    missing = [name for name, value in config.items() if not value]
    if missing:
        raise SalesforceAuthError(
            f"Variables d'environnement manquantes : {', '.join(missing)}. "
            "Voir .env.example."
        )
    return config


# reraise=True : sans ce paramètre, tenacity enveloppe l'exception
# d'origine dans un tenacity.RetryError une fois les tentatives épuisées —
# un type que rien en aval ne rattrape (run.py n'attrape que
# SalesforceAuthError), donc un souci réseau persistant plantait avec une
# trace brute au lieu du message d'erreur propre voulu ici. Bug trouvé en
# écrivant le noeud d'enrichissement (voir enrichment.py, même correctif).
@retry(
    retry=_RETRYABLE, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True
)
def _post_token_request(url: str, data: dict) -> requests.Response:
    return requests.post(url, data=data, timeout=10)


def _get_token_client_credentials() -> tuple[str, str]:
    config = _require_env("SF_CLIENT_ID", "SF_CLIENT_SECRET", "SF_LOGIN_URL")
    response = _post_token_request(
        f"{config['SF_LOGIN_URL']}/services/oauth2/token",
        {
            "grant_type": "client_credentials",
            "client_id": config["SF_CLIENT_ID"],
            "client_secret": config["SF_CLIENT_SECRET"],
        },
    )
    if response.status_code != 200:
        raise SalesforceAuthError(
            f"Échec d'authentification (client_credentials) : {response.text}"
        )
    payload = response.json()
    return payload["access_token"], payload["instance_url"]


def _build_jwt_assertion(config: dict) -> str:
    with open(config["SF_PRIVATE_KEY_PATH"], "rb") as f:
        private_key = f.read()

    now = int(time.time())
    claims = {
        "iss": config["SF_CLIENT_ID"],
        "sub": config["SF_USERNAME"],
        "aud": config["SF_LOGIN_URL"],
        "exp": now + 300,
    }
    # RS256 : signé avec la clé privée, vérifié côté Salesforce avec le
    # certificat public uploadé sur l'External Client App.
    return jwt.encode(claims, private_key, algorithm="RS256")


def _get_token_jwt() -> tuple[str, str]:
    config = _require_env("SF_CLIENT_ID", "SF_USERNAME", "SF_LOGIN_URL", "SF_PRIVATE_KEY_PATH")
    assertion = _build_jwt_assertion(config)
    response = _post_token_request(
        f"{config['SF_LOGIN_URL']}/services/oauth2/token",
        {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": assertion,
        },
    )
    if response.status_code != 200:
        raise SalesforceAuthError(f"Échec d'authentification (JWT) : {response.text}")
    payload = response.json()
    return payload["access_token"], payload["instance_url"]


def get_access_token() -> tuple[str, str]:
    """Renvoie (access_token, instance_url), selon SF_AUTH_FLOW (par défaut
    'client_credentials')."""
    flow = os.getenv("SF_AUTH_FLOW", "client_credentials").lower()
    if flow == "client_credentials":
        return _get_token_client_credentials()
    if flow == "jwt":
        return _get_token_jwt()
    raise SalesforceAuthError(
        f"SF_AUTH_FLOW={flow!r} inconnu (attendu 'client_credentials' ou 'jwt')"
    )


@retry(
    retry=_RETRYABLE, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True
)
def _query_page(access_token: str, url: str) -> dict:
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    if response.status_code == 401:
        # Ne sert à rien de retenter avec le même token expiré : on le
        # signale clairement plutôt que d'épuiser les tentatives pour rien.
        raise SalesforceAuthError("Token expiré ou invalide (401) — relancer l'authentification.")
    response.raise_for_status()
    return response.json()


def _query_all(instance_url: str, access_token: str, soql: str) -> list[dict]:
    """SOQL peut renvoyer les résultats sur plusieurs pages (2000
    enregistrements par défaut) : on suit `nextRecordsUrl` jusqu'au bout."""
    url = f"{instance_url}/services/data/{API_VERSION}/query?q={quote(soql)}"
    records: list[dict] = []
    while url:
        page = _query_page(access_token, url)
        records.extend(page.get("records", []))
        next_path = page.get("nextRecordsUrl")
        url = f"{instance_url}{next_path}" if next_path and not page.get("done", True) else None
    return records


def _contact_to_dict(record: dict) -> dict:
    account = record.get("Account") or {}
    return {
        "id": record["Id"],
        "first_name": record.get("FirstName") or "",
        "last_name": record.get("LastName") or "",
        "email": record.get("Email") or "",
        "phone": record.get("Phone") or "",
        "company": account.get("Name") or "",
        "city": record.get("MailingCity") or "",
        "object_type": "Contact",
        "duplicate_of": None,  # vérité terrain inconnue sur de vraies données
    }


def _lead_to_dict(record: dict) -> dict:
    return {
        "id": record["Id"],
        "first_name": record.get("FirstName") or "",
        "last_name": record.get("LastName") or "",
        "email": record.get("Email") or "",
        "phone": record.get("Phone") or "",
        "company": record.get("Company") or "",
        "city": record.get("City") or "",
        "object_type": "Lead",
        "duplicate_of": None,
    }


def fetch_contacts_and_leads(limit: int | None = 500) -> list[dict]:
    """Point d'entrée : authentifie puis récupère Contacts + Leads, dans le
    même format que `mock_data.generate_contacts()`. `limit` évite de
    ramener tout un org de prod par erreur pendant les tests du POC."""
    access_token, instance_url = get_access_token()

    limit_clause = f" LIMIT {limit}" if limit else ""
    contact_soql = (
        "SELECT Id, FirstName, LastName, Email, Phone, Account.Name, MailingCity "
        "FROM Contact" + limit_clause
    )
    lead_soql = (
        "SELECT Id, FirstName, LastName, Email, Phone, Company, City "
        "FROM Lead WHERE IsConverted = false" + limit_clause
    )

    contacts = [_contact_to_dict(r) for r in _query_all(instance_url, access_token, contact_soql)]
    leads = [_lead_to_dict(r) for r in _query_all(instance_url, access_token, lead_soql)]

    return contacts + leads


if __name__ == "__main__":
    records = fetch_contacts_and_leads(limit=50)
    print(f"{len(records)} enregistrements récupérés (Contact + Lead).")
    for r in records[:10]:
        print(r)
