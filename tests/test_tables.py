"""Structured-table lookup must preserve qualifiers and cell-level uncertainty."""

from evidence_lab.tables import lookup, normalize_docling, verify_reviewed_answer


def fact(*, row_id="AB-12", column="Capacity", value="42", unit="MW",
         conditions=None, provenance=None):
    if conditions is None:
        conditions = {"temperature": "25 C"}
    if provenance is None:
        provenance = {
            "doc_id": "spec.pdf", "page": 3, "table_id": "t1",
            "cell_bbox": [100, 200, 150, 220],
        }
    return {
        "row_id": row_id, "column": column, "value": value, "unit": unit,
        "conditions": conditions, "provenance": provenance,
        "source_verified": False,
    }


def tables(*facts):
    return [{"table_id": "t1", "facts": list(facts)}]


def test_exact_identifier_and_qualifiers_select_one_cell():
    result = lookup(tables(fact(), fact(row_id="AB-123", value="99")),
                    " AB-12 ", " capacity ",
                    conditions={"temperature": "25 C"}, unit="MW")
    assert result["status"] == "answer"
    assert result["value"] == "42"
    assert result["unit"] == "MW"
    assert result["provenance"]["page"] == 3
    assert result["provenance"]["cell_bbox"] == [100, 200, 150, 220]
    assert result["source_verified"] is False


def test_wrong_identifier_never_falls_back_to_similar_row():
    result = lookup(tables(fact()), "AB-1", "Capacity",
                    conditions={"temperature": "25 C"}, unit="MW")
    assert result["status"] == "abstain"
    assert result.get("value") is None


def test_missing_condition_or_unit_does_not_guess():
    qualified = tables(fact(value="42", conditions={"temperature": "25 C"}),
                       fact(value="38", conditions={"temperature": "40 C"}))
    assert lookup(qualified, "AB-12", "Capacity", unit="MW")["status"] == "abstain"
    wrong_condition = lookup(qualified, "AB-12", "Capacity",
                             conditions={"temperature": "30 C"}, unit="MW")
    assert wrong_condition["status"] == "abstain"
    assert wrong_condition.get("value") is None
    wrong_unit = lookup(tables(fact()), "AB-12", "Capacity",
                        conditions={"temperature": "25 C"}, unit="kW")
    assert wrong_unit["status"] == "abstain"
    assert wrong_unit.get("value") is None
    mixed_units = tables(fact(), fact(value="42000", unit="kW"))
    assert lookup(mixed_units, "AB-12", "Capacity",
                  conditions={"temperature": "25 C"})["status"] == "abstain"


def test_duplicate_matching_rows_abstain_even_if_values_agree():
    result = lookup(tables(fact(), fact()), "AB-12", "Capacity",
                    conditions={"temperature": "25 C"}, unit="MW")
    assert result["status"] == "abstain"
    assert result.get("value") is None


def test_missing_cell_provenance_abstains_and_locator_is_not_verification():
    incomplete = fact(provenance={"doc_id": "spec.pdf", "page": 3,
                                  "table_id": "t1"})
    result = lookup(tables(incomplete), "AB-12", "Capacity",
                    conditions={"temperature": "25 C"}, unit="MW")
    assert result["status"] == "abstain"
    assert result.get("value") is None


def test_no_numeric_inference_from_neighboring_cells():
    result = lookup(tables(fact(column="Input", value="21"),
                           fact(column="Output", value="2")),
                    "AB-12", "Difference", conditions={"temperature": "25 C"},
                    unit="MW")
    assert result["status"] == "abstain"
    assert result.get("value") is None


def test_docling_transpose_keeps_value_cell_geometry_and_unverified_flag():
    def cell(row, col, text, bbox):
        return {"start_row_offset_idx": row, "end_row_offset_idx": row + 1,
                "start_col_offset_idx": col, "end_col_offset_idx": col + 1,
                "text": text, "bbox": bbox}

    document = {"tables": [{"self_ref": "#/tables/0", "prov": [
        {"page_no": 2, "bbox": [1, 2, 200, 300]}], "data": {
        "num_rows": 2, "num_cols": 3, "table_cells": [
            cell(0, 0, "Specification", [1, 2, 20, 12]),
            cell(0, 1, "M270TF-XX", [21, 2, 80, 12]),
            cell(0, 2, "M320TF-XX", [81, 2, 140, 12]),
            cell(1, 0, "Product Weight", [1, 13, 20, 23]),
            cell(1, 1, "9.5 Kg", [21, 13, 80, 23]),
            cell(1, 2, "11 Kg", [81, 13, 140, 23]),
        ]}}]}
    extracted = normalize_docling(document, "manual.pdf")
    result = lookup(extracted, "M320TF-XX", "Product Weight", unit="kg")
    assert result["status"] == "answer"
    assert result["value"] == 11.0
    assert result["provenance"]["doc_id"] == "manual.pdf"
    assert result["provenance"]["page"] == 2
    assert result["provenance"]["cell_bbox"] == [81, 13, 140, 23]
    assert result["source_verified"] is False


def test_review_verification_requires_matching_source_hash_and_unique_review():
    extracted = fact(value=9.5, unit="kg", conditions={})
    extracted["provenance"].update(source_sha256="a" * 64,
                                   row_id="AB-12", field="Capacity")
    answer = lookup(tables(extracted), "AB-12", "Capacity", unit="kg")
    review = {"review_id": "review-1", "method": "human PDF inspection",
              "doc_id": "spec.pdf", "page": 3, "source_sha256": "a" * 64,
              "row_id": "AB-12", "field": "Capacity", "conditions": {},
              "value": 9.5, "unit": "kg"}
    assert verify_reviewed_answer(answer, [review])["source_verified"] is True
    wrong_hash = {**review, "source_sha256": "b" * 64}
    assert verify_reviewed_answer(answer, [wrong_hash])["source_verified"] is False
    wrong_value = {**review, "value": 11.0}
    assert verify_reviewed_answer(answer, [wrong_value])["source_verified"] is False
    assert verify_reviewed_answer(answer, [review, review])["source_verified"] is False
