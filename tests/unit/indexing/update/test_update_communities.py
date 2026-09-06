# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for incremental community id reconciliation."""

import pandas as pd
from graphrag.index.update.communities import _update_and_merge_communities
from graphrag.index.update.relationships import _update_and_merge_relationships


def _community_row(
    *,
    row_id: str,
    community: int,
    entity_ids: list[str] | None,
    relationship_ids: list[str] | None,
    parent: int = -1,
) -> dict:
    return {
        "id": row_id,
        "human_readable_id": community,
        "community": community,
        "level": 0,
        "parent": parent,
        "children": [],
        "title": f"Community {community}",
        "entity_ids": entity_ids,
        "relationship_ids": relationship_ids,
        "text_unit_ids": ["tu1"],
        "period": None,
        "size": len(entity_ids) if entity_ids is not None else None,
    }


def _relationship_row(
    *,
    relationship_id: str,
    source: str,
    target: str,
    human_readable_id: int,
) -> dict:
    return {
        "id": relationship_id,
        "human_readable_id": human_readable_id,
        "source": source,
        "target": target,
        "description": f"{source}->{target}",
        "weight": 1.0,
        "combined_degree": 2,
        "text_unit_ids": ["tu1"],
    }


def test_update_and_merge_relationships_returns_surviving_id_mapping():
    """Overlapping delta relationships should map to the surviving old id."""
    old_relationships = pd.DataFrame([
        _relationship_row(
            relationship_id="r-old", source="ALPHA", target="BETA", human_readable_id=0
        )
    ])
    delta_relationships = pd.DataFrame([
        _relationship_row(
            relationship_id="r-delta",
            source="ALPHA",
            target="BETA",
            human_readable_id=0,
        )
    ])

    merged_relationships, relationship_id_mapping = _update_and_merge_relationships(
        old_relationships, delta_relationships
    )

    assert len(merged_relationships) == 1
    assert relationship_id_mapping == {"r-delta": "r-old"}


def test_update_and_merge_communities_rewrites_entity_ids_with_surviving_mapping():
    """Delta community members should point at surviving entity ids after merge."""
    old_communities = pd.DataFrame([
        _community_row(
            row_id="c-old",
            community=0,
            entity_ids=["e-1", "e-2"],
            relationship_ids=["r-old"],
        )
    ])
    delta_communities = pd.DataFrame([
        _community_row(
            row_id="c-delta",
            community=0,
            entity_ids=["d-1", "d-2"],
            relationship_ids=["r-delta"],
        )
    ])

    merged_communities, _ = _update_and_merge_communities(
        old_communities,
        delta_communities,
        entity_id_mapping={"d-1": "e-1"},
        relationship_id_mapping={},
    )

    delta_row = merged_communities.loc[merged_communities["id"] == "c-delta"].iloc[0]
    assert delta_row["entity_ids"] == ["e-1", "d-2"]


def test_update_and_merge_communities_rewrites_relationship_ids_with_surviving_mapping():
    """Delta community relationships should point at surviving relationship ids after merge."""
    old_communities = pd.DataFrame([
        _community_row(
            row_id="c-old",
            community=0,
            entity_ids=["e-1", "e-2"],
            relationship_ids=["r-old"],
        )
    ])
    delta_communities = pd.DataFrame([
        _community_row(
            row_id="c-delta",
            community=0,
            entity_ids=["e-1", "d-2"],
            relationship_ids=["r-delta", "r-fresh"],
        )
    ])

    merged_communities, _ = _update_and_merge_communities(
        old_communities,
        delta_communities,
        entity_id_mapping={},
        relationship_id_mapping={"r-delta": "r-old"},
    )

    delta_row = merged_communities.loc[merged_communities["id"] == "c-delta"].iloc[0]
    assert delta_row["relationship_ids"] == ["r-old", "r-fresh"]
