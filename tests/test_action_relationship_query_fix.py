from vidquery.search import StructuredQueryParser


def test_typing_maps_to_work_on_computer() -> None:
    plan = StructuredQueryParser().parse("when is someone typing?")
    assert plan.intent == "action_search"
    assert plan.actions == ["work on computer"]
    assert plan.spoken_terms == []


def test_people_interacting_maps_to_talk_to() -> None:
    plan = StructuredQueryParser().parse("when are two people interacting?")
    assert plan.intent == "relationship_search"
    assert any(
        item.predicate == "talk_to" and item.subject == "person" and item.object == "person"
        for item in plan.relationship_tuples
    )
    assert "interacting" not in plan.spoken_terms


def test_talking_to_another_person_is_relationship_not_spoken() -> None:
    plan = StructuredQueryParser().parse("someone talking to another person")
    assert plan.relationship_tuples
    assert plan.relationship_tuples[0].predicate == "talk_to"
    assert "another" not in plan.spoken_terms
    assert plan.intent == "relationship_search"
