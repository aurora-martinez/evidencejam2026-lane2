
from search import (
    contains_term,
    match_group,
    expand_synonyms,
    parse_question,
)


def test_plural_matching():
    assert contains_term("seedlings establish quickly", "seedling")


def test_no_partial_word_matching():
    assert not contains_term("price increased", "ice")


def test_exact_word_matching():
    assert contains_term("native plant cover", "native")


def test_synonyms():
    synonyms = expand_synonyms("invasive")

    assert "non-native" in synonyms
    assert "alien" in synonyms


def test_and_matching():
    node = {
        "name": "Wildfire in boreal forests",
        "entity_term": "wildfire",
        "measured_attribute": "",
        "embedding_text": "",
    }

    assert match_group(node, ["wildfire", "boreal"]) > 0
    assert match_group(node, ["wildfire", "grassland"]) == 0


def test_field_specific_matching():
    node = {
        "name": "Invasive plant abundance",
        "entity_term": "invasive plant",
        "measured_attribute": "abundance",
        "embedding_text": "",
    }

    group = {
        "entity_term": ["invasive"],
        "measured_attribute": ["abundance"],
    }

    assert match_group(node, group) > 0


def test_question_parser():
    plan = parse_question(
        "How does nursery fertilization affect seedling growth?"
    )

    assert plan is not None
    assert plan["interventions"] == ["nursery fertilization"]
    assert plan["outcomes"] == ["seedling growth"]


def test_unsupported_question():
    assert parse_question(
        "Which species establish in Ontario?"
    ) is None
