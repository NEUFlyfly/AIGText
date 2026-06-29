import json
from pathlib import Path
from typing import TypeAlias, cast

import pytest


REQUIRED_KEYS = {
    "doc_id",
    "coarse_category",
    "sub_category",
    "aliases",
    "description",
    "image_dir",
    "document_path",
}

REQUIRED_IDENTITY_KEYS = ("doc_id", "coarse_category", "sub_category")

EXPECTED_FIXTURES = {
    "fixture_temp_sensor": ("智能传感器", "温湿度传感器"),
    "fixture_pir_sensor": ("智能传感器", "人体红外传感器"),
    "fixture_ptz_camera": ("智能摄像头", "室内云台摄像头"),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = REPO_ROOT / "data" / "iot_knowledge" / "taxonomy.json"
TaxonomyEntry: TypeAlias = dict[str, object]


def validate_taxonomy_entry(entry: TaxonomyEntry) -> None:
    for required_key in REQUIRED_IDENTITY_KEYS:
        if required_key not in entry:
            raise AssertionError(f"taxonomy entry missing required key: {required_key}")

    missing_keys = REQUIRED_KEYS - set(entry)
    if missing_keys:
        missing_key_list = ", ".join(sorted(missing_keys))
        raise AssertionError(f"taxonomy entry missing required keys: {missing_key_list}")

    aliases = entry["aliases"]
    description = entry["description"]
    image_dir = entry["image_dir"]
    document_path = entry["document_path"]

    assert isinstance(aliases, list), "taxonomy entry aliases must be a list"
    assert isinstance(description, str), "taxonomy entry description must be a string"
    assert isinstance(image_dir, str), "image_dir must be a string"
    assert isinstance(document_path, str), "document_path must be a string"
    assert not Path(image_dir).is_absolute(), "image_dir must be a relative path"
    assert not Path(document_path).is_absolute(), "document_path must be a relative path"


def load_taxonomy() -> list[TaxonomyEntry]:
    with TAXONOMY_PATH.open("r", encoding="utf-8") as taxonomy_file:
        taxonomy_data = cast(object, json.load(taxonomy_file))

    assert isinstance(taxonomy_data, list), "taxonomy must be an array of objects"
    taxonomy_entries = cast(list[object], taxonomy_data)
    for entry in taxonomy_entries:
        assert isinstance(entry, dict), "taxonomy must be an array of objects"

    return [cast(TaxonomyEntry, entry) for entry in taxonomy_entries]


def test_fixture_taxonomy_entries_are_valid():
    taxonomy_entries = load_taxonomy()

    assert isinstance(taxonomy_entries, list), "taxonomy must be an array of objects"
    assert len(taxonomy_entries) == 3

    entries_by_doc_id: dict[str, TaxonomyEntry] = {}
    for entry in taxonomy_entries:
        validate_taxonomy_entry(entry)
        doc_id = entry["doc_id"]
        assert isinstance(doc_id, str), "taxonomy entry doc_id must be a string"
        entries_by_doc_id[doc_id] = entry

    assert set(entries_by_doc_id) == set(EXPECTED_FIXTURES)

    for doc_id, expected_categories in EXPECTED_FIXTURES.items():
        entry = entries_by_doc_id[doc_id]
        coarse_category, sub_category = expected_categories
        assert entry["coarse_category"] == coarse_category
        assert entry["sub_category"] == sub_category
        document_path = entry["document_path"]
        image_dir = entry["image_dir"]
        assert isinstance(document_path, str)
        assert isinstance(image_dir, str)
        assert (REPO_ROOT / document_path).is_file()
        assert (REPO_ROOT / image_dir).is_dir()


@pytest.mark.parametrize("missing_key", REQUIRED_IDENTITY_KEYS)
def test_missing_identity_keys_fail_with_explicit_messages(missing_key: str) -> None:
    valid_entry: TaxonomyEntry = {
        "doc_id": "fixture_temp_sensor",
        "coarse_category": "智能传感器",
        "sub_category": "温湿度传感器",
        "aliases": ["温湿度探头"],
        "description": "fixture description",
        "image_dir": "tests/fixtures/iot_knowledge/智能传感器/温湿度传感器/images",
        "document_path": "tests/fixtures/iot_knowledge/智能传感器/温湿度传感器/document.md",
    }
    invalid_entry = dict(valid_entry)
    _ = invalid_entry.pop(missing_key)

    with pytest.raises(AssertionError, match=f"taxonomy entry missing required key: {missing_key}"):
        validate_taxonomy_entry(invalid_entry)
