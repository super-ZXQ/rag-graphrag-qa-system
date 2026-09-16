"""评测接线：提供 graph_search_factory 供 run_retrieval 自动拾取。"""
from __future__ import annotations

from graph.expansion import GraphExpander
from graph.store import get_graph


def graph_search_factory(retriever, store=None):
    graph = get_graph()
    expander = GraphExpander(retriever=retriever, graph=graph)

    def search(query: str) -> list[dict]:
        return expander.search(query)

    return search
