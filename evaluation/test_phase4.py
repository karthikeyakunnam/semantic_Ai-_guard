import csv, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.services.chunking import chunk_passages
from app.services.hybrid_similarity import classify_passages, overlap_signals, hybrid_score
from hybrid_evaluation import validate_disjoint_splits, calibrate, evaluate, freeze_config
from phase4_benchmark import read_rows
import phase4_benchmark
import numpy as np

def test_passage_offsets_reconstruct_original_span():
    source = "  First extracted passage has enough content to be retained exactly.\nSecond extracted passage also preserves its original location.  "
    chunks = chunk_passages(source)
    assert chunks
    for p in chunks:
        assert source[p.start_char:p.end_char].strip() == p.text
        assert 0 <= p.start_char < p.end_char <= len(source)

def test_filter_marks_but_does_not_discard_reference_and_boilerplate():
    source = "Assignment prompt: discuss the following question in your own words.\nReferences\nAuthor, 2020. A sufficiently detailed bibliographic entry for audit purposes."
    chunks = chunk_passages(source)
    marked = classify_passages(chunks, ["discuss the following question"])
    excluded = [p for p in marked if p.excluded]
    assert len(excluded) >= 3
    assert all(source[p.start_char:p.end_char].strip() == p.text for p in excluded)
    assert {p.exclusion_reason for p in excluded} == {"configured_boilerplate", "reference_section"}

def test_short_extraction_fragment_is_audited_but_not_candidate():
    source = "A short PDF line fragment.\nThis longer passage contains enough detail to remain eligible for comparison."
    marked = classify_passages(chunk_passages(source))
    assert marked[0].excluded and marked[0].exclusion_reason == "insufficient_length"
    assert source[marked[0].start_char:marked[0].end_char] == marked[0].text
    assert not marked[1].excluded

def test_signals_keep_semantic_lexical_phrase_and_partial_distinct():
    signals = overlap_signals("A unique password protects an account from unauthorized access", "A unique password protects an account while recovery settings add safety")
    assert 0 < signals["lexical_similarity"] < 1
    assert 0 < signals["phrase_overlap"] <= 1
    assert signals["partial_copy_signal"] > 0
    score = hybrid_score(.7, signals, {"semantic": .5, "lexical": .25, "phrase": .25})
    assert 0 <= score <= 1

def test_calibration_uses_validation_rows_and_freezes_selected_configuration(tmp_path):
    rows = [
        {"semantic": .9, "lexical": .9, "phrase": .9, "relevant": 1},
        {"semantic": .1, "lexical": .0, "phrase": .0, "relevant": 0},
    ]
    config = calibrate(rows, weight_grid=[{"semantic": .5, "lexical": .25, "phrase": .25}], thresholds=[.5])
    val_path = tmp_path / "validation.csv"; val_path.write_text("synthetic validation only")
    frozen = freeze_config(tmp_path / "frozen.json", config, val_path)
    assert frozen["frozen"] is True
    assert frozen["calibration_source"] == "validation_only"
    assert evaluate(rows, frozen)["f1"] == 1

def test_split_validation_rejects_topic_or_template_leakage():
    val = [{"topic": "topic-a", "template": "template-a"}]
    test = [{"topic": "topic-b", "template": "template-b"}]
    validate_disjoint_splits(val, test)
    with pytest.raises(ValueError, match="topic leakage"):
        validate_disjoint_splits(val, [{"topic": "topic-a", "template": "template-b"}])
    with pytest.raises(ValueError, match="template leakage"):
        validate_disjoint_splits(val, [{"topic": "topic-b", "template": "template-a"}])

def test_false_positive_rate_counts_independent_common_topic_negatives():
    rows = [
        {"semantic": .9, "lexical": .2, "phrase": .1, "relevant": 0, "scenario": "common_topic_independent"},
        {"semantic": .8, "lexical": .9, "phrase": .9, "relevant": 1, "scenario": "exact_copy"},
    ]
    config = {"weights": {"semantic": .5, "lexical": .25, "phrase": .25}, "threshold": .5}
    metrics = evaluate(rows, config)
    assert metrics["fp"] == 1 and metrics["tn"] == 0
    assert metrics["false_positive_rate"] == 1
    assert metrics["scores"][0] > config["threshold"]

def test_phase4_datasets_disjoint_and_output_report_generation_contract():
    val = read_rows(ROOT / "evaluation/datasets/hybrid_validation.csv")
    test = read_rows(ROOT / "evaluation/datasets/hybrid_test.csv")
    validate_disjoint_splits(val, test)
    assert {r["scenario"] for r in val} >= {"Exact copying", "Paraphrase", "Partial copy", "Common topic independent"}
    assert {r["scenario"] for r in test} >= {"Exact copying", "Paraphrase", "Partial copy", "Common topic independent"}

def test_phase4_benchmark_generates_frozen_config_csv_json_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(phase4_benchmark, "get_model", lambda: object())
    monkeypatch.setattr(phase4_benchmark, "get_reranker", lambda: object())
    def fake_encode(texts):
        values=[]
        for text in texts:
            seed=sum(text.encode("utf-8"))
            vector=np.array([seed % 97 + 1, len(text) % 31 + 1, 1], dtype=np.float32)
            values.append(vector / np.linalg.norm(vector))
        return np.stack(values)
    monkeypatch.setattr(phase4_benchmark, "encode", fake_encode)
    monkeypatch.setattr(phase4_benchmark, "rerank", lambda pairs: [.5] * len(pairs))
    output=tmp_path/"results"
    phase4_benchmark.run(ROOT/"evaluation/datasets/hybrid_validation.csv", ROOT/"evaluation/datasets/hybrid_test.csv", output)
    assert (output/"hybrid_calibration.json").exists()
    assert (output/"hybrid_test_pairs.csv").exists()
    assert (output/"phase4_metrics.json").exists()
    report=(output/"phase4_report.md").read_text(encoding="utf-8")
    assert "Held-out test metrics" in report
    assert "Scenario breakdown" in report
    assert "not real-world plagiarism accuracy" in report
