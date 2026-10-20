"""Point d'entrée : charge des contacts (mock ou vrai sandbox Salesforce),
fait tourner le graphe jusqu'au bout — y compris la pause
"human-in-the-loop" quand le graphe l'exige — et affiche le résultat final
(jugements, merges/conversions, journal de monitoring).

    uv run python -m sf_dedup_poc.run                    # données mock (par défaut)
    uv run python -m sf_dedup_poc.run --source salesforce # vrai sandbox (voir README)
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langgraph.types import Command

from .graph import build_graph
from .mock_data import generate_contacts

# Charge .env s'il existe (sans erreur sinon) : c'est de là que
# salesforce_source lit SF_CLIENT_ID etc. quand --source salesforce est
# utilisé. N'a aucun effet en mode mock.
load_dotenv()


def _fmt_contact(c: dict) -> str:
    return f"{c['first_name']} {c['last_name']} <{c['email']}> {c['phone']}"


def _ask_human(item: dict) -> bool:
    """Simule la validation humaine dans le terminal. Dans un vrai système,
    ce serait un bouton "Approuver" dans Slack ou dans une UI Salesforce —
    le node `human_review` n'a besoin de rien savoir de plus que "je reçois
    un dict de décisions au resume", donc cette fonction pourrait être
    remplacée sans toucher au graphe."""
    prompt = (
        f"\n[REVUE HUMAINE] score={item['score']}  confiance LLM/heuristique={item['confidence']:.2f}\n"
        f"  raison : {item['reason']}\n"
        f"  A: {_fmt_contact(item['contact_a'])}\n"
        f"  B: {_fmt_contact(item['contact_b'])}\n"
        "  Fusionner ces deux fiches ? [o/N] "
    )
    try:
        answer = input(prompt)
    except EOFError:
        print("  (pas de terminal interactif détecté -> rejet par défaut)")
        return False
    return answer.strip().lower() in {"o", "oui", "y", "yes"}


def _load_contacts(source: str, limit: int) -> tuple[list[dict], bool]:
    """Renvoie (contacts, a_une_verite_terrain). La vérité terrain
    (duplicate_of connu) n'existe que pour les données mock — sur un vrai
    sandbox on ne peut valider les résultats qu'à l'oeil / via la revue
    humaine."""
    if source == "mock":
        contacts = generate_contacts(n_base=40, dup_rate=0.5, seed=42)
        return [c.to_dict() for c in contacts], True

    from .salesforce_source import SalesforceAuthError, fetch_contacts_and_leads

    try:
        records = fetch_contacts_and_leads(limit=limit)
    except SalesforceAuthError as exc:
        raise SystemExit(f"Impossible de récupérer les données Salesforce : {exc}") from exc
    return records, False


def main() -> None:
    parser = argparse.ArgumentParser(description="POC dédoublonnage Salesforce avec LangGraph")
    parser.add_argument(
        "--source", choices=["mock", "salesforce"], default="mock",
        help="mock (par défaut) = données synthétiques ; salesforce = vrai sandbox (voir README)",
    )
    parser.add_argument(
        "--limit", type=int, default=200,
        help="Nombre max de Contact/Lead à récupérer par objet (source=salesforce uniquement)",
    )
    args = parser.parse_args()

    raw_contacts, has_ground_truth = _load_contacts(args.source, args.limit)
    by_id = {c["id"]: c for c in raw_contacts}

    app = build_graph()
    config = {"configurable": {"thread_id": f"sf-dedup-demo-{args.source}"}}

    result = app.invoke({"raw_contacts": raw_contacts}, config=config)

    # Tant que le graphe est en pause sur human_review, __interrupt__ est
    # présent dans le state renvoyé. On boucle : on demande les décisions
    # humaines, on relance avec Command(resume=...), et on recommence s'il y
    # a encore une pause (cas rare ici, mais réaliste si plusieurs vagues de
    # revue sont nécessaires).
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        decisions = {
            f"{item['id_a']}:{item['id_b']}": _ask_human(item)
            for item in payload["pending_reviews"]
        }
        result = app.invoke(Command(resume=decisions), config=config)

    print(f"\n{len(raw_contacts)} enregistrements chargés ({args.source}), "
          f"{len(result['judgments'])} paires évaluées.\n")

    # Toujours affiché, même quand rien n'est fusionné/revu : sinon, sur un
    # sandbox où la plupart des paires sont jugées "pas doublon", on ne voit
    # littéralement rien d'autre que les compteurs agrégés du log — pas très
    # utile pour comprendre CE qui a été évalué et pourquoi.
    print("--- Détail des jugements ---")
    for j in result.get("judgments", []):
        a, b = by_id[j["id_a"]], by_id[j["id_b"]]
        print(
            f"[{'DOUBLON' if j['is_duplicate'] else 'pas doublon'}] "
            f"score={j['score']:5.1f}  confiance={j['confidence']:.2f}  source={j['source']}"
        )
        print(f"    A: {_fmt_contact(a)}")
        print(f"    B: {_fmt_contact(b)}")
        print(f"    raison: {j['reason']}")

    print()
    for line in result.get("log", []):
        print(line)

    print("\n--- Détail des rapprochements ---")
    for m in result.get("merges", []):
        survivor, absorbed = by_id[m["survivor_id"]], by_id[m["absorbed_id"]]
        label = "MERGE" if m["kind"] == "merge" else "CONVERSION CANDIDATE"
        print(
            f"[{label}] {absorbed['id']} ({absorbed.get('object_type', 'Contact')}) -> "
            f"{m['survivor_id']} ({survivor.get('object_type', 'Contact')})  [source={m['source']}]"
        )
        print(f"    gardé  : {_fmt_contact(survivor)}")
        print(f"    absorbé: {_fmt_contact(absorbed)}")

    print("\n--- Détail des enrichissements ---")
    for contact_id, e in result.get("enrichments", {}).items():
        c = by_id[contact_id]
        if e["source"] in ("skipped-no-email", "skipped-generic-domain"):
            print(f"  [ignoré: {e['source']}] {_fmt_contact(c)}")
            continue
        mx_label = {True: "MX ok", False: "MX ABSENT (domaine suspect)", None: "non vérifié"}[e["mx_valid"]]
        print(f"  [{mx_label}] {_fmt_contact(c)}  -> entreprise devinée: {e['guessed_company']}")

    if has_ground_truth:
        n_true_dups = sum(1 for c in raw_contacts if c.get("duplicate_of") is not None)
        merged_absorbed_ids = {m["absorbed_id"] for m in result.get("merges", [])}
        true_positive = sum(
            1
            for c in raw_contacts
            if c.get("duplicate_of") is not None and c["id"] in merged_absorbed_ids
        )
        print("\n--- Bilan (contre la vérité terrain connue du générateur mock) ---")
        print(f"Vrais doublons injectés : {n_true_dups}")
        print(f"Correctement fusionnés  : {true_positive}")
        print(f"Total de rapprochements décidés : {len(result.get('merges', []))}")


if __name__ == "__main__":
    main()
