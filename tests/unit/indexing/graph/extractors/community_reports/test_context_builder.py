# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Regression tests for _prepare_reports_at_level edge aggregation.

See https://github.com/microsoft/graphrag/issues/2583 — the previous
groupby(...).agg(EDGE_DETAILS="first") discarded every edge except the
first one per source and per target, so a triangle (A→B, A→C, C→B)
lost A→C.
"""

import graphrag.data_model.schemas as schemas
import pandas as pd
from graphrag.index.operations.summarize_communities.graph_context.context_builder import (
    _prepare_reports_at_level,
)


class _FakeTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()

    def num_tokens(self, text: str) -> int:
        return len(text.split())


def _make_nodes(titles: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        schemas.TITLE: titles,
        schemas.SHORT_ID: range(len(titles)),
        schemas.DESCRIPTION: ["desc"] * len(titles),
        schemas.NODE_DEGREE: [0] * len(titles),
        schemas.COMMUNITY_ID: [0] * len(titles),
        schemas.COMMUNITY_LEVEL: [0] * len(titles),
    }).assign(**{
        schemas.NODE_DETAILS: lambda df: df[
            [schemas.SHORT_ID, schemas.TITLE, schemas.DESCRIPTION, schemas.NODE_DEGREE]
        ].to_dict(orient="records")
    })


def _make_edges(sources: list[str], targets: list[str]) -> pd.DataFrame:
    n = len(sources)
    df = pd.DataFrame({
        schemas.EDGE_SOURCE: sources,
        schemas.EDGE_TARGET: targets,
        schemas.SHORT_ID: range(n),
        schemas.DESCRIPTION: ["e"] * n,
        schemas.EDGE_DEGREE: [1] * n,
    })
    df[schemas.EDGE_DETAILS] = df[
        [
            schemas.SHORT_ID,
            schemas.EDGE_SOURCE,
            schemas.EDGE_TARGET,
            schemas.DESCRIPTION,
            schemas.EDGE_DEGREE,
        ]
    ].to_dict(orient="records")
    return df


def _edge_ids_in_context(result: pd.DataFrame) -> set[int]:
    ctx = result.iloc[0][schemas.CONTEXT_STRING]
    if "-----Relationships-----" not in ctx:
        return set()
    rel_section = ctx.split("-----Relationships-----")[1].strip()
    ids = set()
    for line in rel_section.split("\n")[1:]:
        parts = line.split(",")
        if parts and parts[0].strip().isdigit():
            ids.add(int(parts[0]))
    return ids


def test_triangle_graph_keeps_all_edges():
    """A→B (id=0), A→C (id=1), C→B (id=2) — previously A→C was dropped."""
    nodes = _make_nodes(["A", "B", "C"])
    edges = _make_edges(["A", "A", "C"], ["B", "C", "B"])
    result = _prepare_reports_at_level(nodes, edges, None, _FakeTokenizer(), 0, 10**9)
    assert _edge_ids_in_context(result) == {0, 1, 2}


def test_chain_graph_keeps_both_directions():
    """A→B (id=0), B→C (id=1) — both must appear regardless of order."""
    nodes = _make_nodes(["A", "B", "C"])
    edges = _make_edges(["A", "B"], ["B", "C"])
    result = _prepare_reports_at_level(nodes, edges, None, _FakeTokenizer(), 0, 10**9)
    assert _edge_ids_in_context(result) == {0, 1}


def test_isolated_node_produces_no_edges():
    """A node with no edges must not crash and must emit no Relationships section."""
    nodes = _make_nodes(["A"])
    edges = pd.DataFrame(
        columns=[
            schemas.EDGE_SOURCE,
            schemas.EDGE_TARGET,
            schemas.SHORT_ID,
            schemas.DESCRIPTION,
            schemas.EDGE_DEGREE,
            schemas.EDGE_DETAILS,
        ]
    )
    result = _prepare_reports_at_level(nodes, edges, None, _FakeTokenizer(), 0, 10**9)
    assert _edge_ids_in_context(result) == set()


def test_self_loop_edge_is_kept():
    """A self-loop edge (A→A) must appear in the context."""
    nodes = _make_nodes(["A"])
    edges = _make_edges(["A"], ["A"])
    result = _prepare_reports_at_level(nodes, edges, None, _FakeTokenizer(), 0, 10**9)
    assert _edge_ids_in_context(result) == {0}
