from evidence_lab.audit import audit_samples


def sample(**changes):
    return {"doc_id": "a.pdf", "question": "Which value?", "answer": "12",
            "answer_format": "Int", "evidence_pages": "[1]",
            "evidence_sources": "['Table']", **changes}


def test_audit_accounts_for_every_row_and_preserves_originals():
    raw = [sample(), sample(question="Bad coordinate", evidence_pages="[0]"),
           sample(question="Missing evidence", evidence_pages="[]"),
           sample(question="Absent", answer="Not answerable", answer_format="None",
                  evidence_pages="[]", evidence_sources="[]")]
    eligible, report = audit_samples(raw, {"a.pdf": 2})
    assert [r["source_row"] for r in eligible] == [0, 3]
    assert report["total_rows"] == report["eligible_rows"] + report["quarantined_rows"] == 4
    assert [r["source_row"] for r in report["quarantine"]] == [1, 2]
    assert "source_row" not in raw[0]


def test_conflicting_duplicate_quarantines_both_without_choosing_answer():
    raw = [sample(), sample(answer="Not answerable", answer_format="None",
                            evidence_pages="[]", evidence_sources="[]")]
    eligible, report = audit_samples(raw, {"a.pdf": 2})
    assert eligible == []
    assert all("conflicting_duplicate" in r["reasons"] for r in report["quarantine"])
