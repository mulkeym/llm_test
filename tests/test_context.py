import pytest
from src.context import ContextGenerator


def test_generate_filler_runbook():
    gen = ContextGenerator()
    text = gen.generate_filler("runbook", target_tokens=500)
    assert len(text) > 100


def test_generate_filler_log_dump():
    gen = ContextGenerator()
    text = gen.generate_filler("log_dump", target_tokens=500)
    assert len(text) > 100


def test_insert_needles_single():
    gen = ContextGenerator()
    filler = "word " * 1000
    needles = [{"text": "SECRET_NEEDLE_123", "position": 0.5}]
    result = gen.insert_needles(filler, needles)
    assert "SECRET_NEEDLE_123" in result


def test_insert_needles_ordering():
    gen = ContextGenerator()
    filler = "word " * 1000
    needles = [
        {"text": "NEEDLE_A", "position": 0.25},
        {"text": "NEEDLE_B", "position": 0.75},
    ]
    result = gen.insert_needles(filler, needles)
    pos_a = result.find("NEEDLE_A")
    pos_b = result.find("NEEDLE_B")
    assert pos_a < pos_b


def test_build_context_document():
    gen = ContextGenerator()
    doc = gen.build_context_document(
        filler_type="runbook",
        target_tokens=500,
        needles=[{"text": "The secret port is 9999", "position": 0.5}],
    )
    assert "The secret port is 9999" in doc
