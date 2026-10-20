"""
Assemblage du graphe LangGraph.

    START -> block_and_match -> judge_pairs -> split_by_confidence
                                                      |
                                    (conditionnel, route_after_split)
                                        /                        \\
                              human_review                    merge_duplicates
                                    \\                        /
                                     -> merge_duplicates ----
                                              |
                                      enrich_contacts -> log_and_monitor -> END

`split_by_confidence` ne fait aucun travail "métier" : c'est un noeud dont
le seul rôle est de préparer la décision de routage pour l'edge
conditionnel juste après (`add_conditional_edges`). Si rien n'est ambigu,
`route_after_split` saute directement à `merge_duplicates` — pas besoin
d'interrompre le graphe pour rien.

`human_review` utilise `interrupt()` : ça nécessite un *checkpointer* pour
que LangGraph puisse suspendre l'exécution et la reprendre plus tard
(potentiellement dans un tout autre process) exactement là où elle s'est
arrêtée. `MemorySaver` est un checkpointer en mémoire, très bien pour un
POC / des tests ; en production on utiliserait un checkpointer persistant
(Postgres, SQLite...) pour survivre à un redémarrage du process pendant
qu'on attend la validation humaine.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    block_and_match,
    enrich_contacts,
    human_review,
    judge_pairs,
    log_and_monitor,
    merge_duplicates,
    route_after_split,
    split_by_confidence,
)
from .state import DedupState


def build_graph():
    graph = StateGraph(DedupState)

    graph.add_node("block_and_match", block_and_match)
    graph.add_node("judge_pairs", judge_pairs)
    graph.add_node("split_by_confidence", split_by_confidence)
    graph.add_node("human_review", human_review)
    graph.add_node("merge_duplicates", merge_duplicates)
    graph.add_node("enrich_contacts", enrich_contacts)
    graph.add_node("log_and_monitor", log_and_monitor)

    graph.add_edge(START, "block_and_match")
    graph.add_edge("block_and_match", "judge_pairs")
    graph.add_edge("judge_pairs", "split_by_confidence")

    graph.add_conditional_edges(
        "split_by_confidence",
        route_after_split,
        {"human_review": "human_review", "merge_duplicates": "merge_duplicates"},
    )

    graph.add_edge("human_review", "merge_duplicates")
    graph.add_edge("merge_duplicates", "enrich_contacts")
    graph.add_edge("enrich_contacts", "log_and_monitor")
    graph.add_edge("log_and_monitor", END)

    return graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    app = build_graph()
    print(app.get_graph().draw_ascii())
