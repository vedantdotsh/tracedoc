from scripts.audit_corpus_accuracy import metrics, score


def test_global_page_match_requires_document_and_page():
    gold = [("right.pdf", 4)]
    ranked = [("wrong.pdf", 4), ("right.pdf", 4)]
    assert score(gold, ranked) == {"recall": 1.0, "hit": 1.0, "mrr": 0.5}


def test_metrics_excludes_unanswerable_and_reads_nested_retrieval():
    cases = [
        {"gold_pages": [4], "retrieval": {"document_bm25": score([("a", 4)], [("a", 4)])}},
        {"gold_pages": [], "retrieval": {}},
        {"gold_pages": [7], "retrieval": {"document_bm25": score([("a", 7)], [])}},
    ]
    assert metrics(cases, "document_bm25") == {
        "positive_questions": 2,
        "mean_page_recall_at_5": 0.5,
        "hit_at_5": 0.5,
        "mrr_at_5": 0.5,
    }
