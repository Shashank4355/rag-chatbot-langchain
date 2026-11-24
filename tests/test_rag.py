"""Unit tests for the parts that decide whether an answer is shown. No API key needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.kb import load_kb, serialize_tables, split_fixed, split_naive  # noqa: E402
from rag.pipeline import REFUSAL, RagService, check_answer  # noqa: E402
from rag.retrieval import codes_in, normalise_codes, tokens  # noqa: E402


@pytest.fixture(scope="module")
def svc():
    import os
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY"):  # extractive mode: deterministic, offline
        os.environ.pop(k, None)
    return RagService()


def test_table_rows_carry_their_column_names():
    out = serialize_tables("| Plan | Price | Seats |\n|---|---|---|\n| Team | $12 | 3 |")
    assert out == "- Team: Price $12; Seats 3."


def test_fixed_chunks_start_with_breadcrumb_and_drop_front_matter():
    chunks = split_fixed(load_kb())
    assert all("\n" in c.page_content for c in chunks)
    assert not any("updated:" in c.page_content for c in chunks)
    price = next(c for c in chunks if c.metadata["section"] == "Price per user")
    assert price.page_content.startswith("Plans and pricing > Price per user")
    assert "Business: Monthly billing $24" in price.page_content


def test_naive_chunks_reproduce_the_usual_problems():
    chunks = split_naive(load_kb())
    assert any("updated:" in c.page_content for c in chunks)  # front matter leaks in
    assert all(c.metadata["section"].startswith("chunk") for c in chunks)  # no idea where a fact lives


@pytest.mark.parametrize("raw", ["kb412", "KB 412", "kb-412", "KB-412"])
def test_error_codes_normalise(raw):
    assert codes_in(normalise_codes(f"I get {raw} on sync")) == {"KB-412"}


def test_ordinary_numbers_are_not_codes():
    assert codes_in("the 260 characters limit and HTTP 429") == set()
    assert "kb-412" in tokens("error KB-412 appeared")


def test_check_flags_a_number_not_in_the_cited_excerpt():
    ex = ["[1] Plans > Price\nBusiness: Monthly billing $24."]
    assert all(c["ok"] for c in check_answer("Business is $24 per user per month [1].", ex, "q"))
    bad = check_answer("Business is $29 per user per month [1].", ex, "q")
    assert not bad[-1]["ok"] and "29" in bad[-1]["detail"]


def test_check_flags_missing_and_dangling_citations():
    ex = ["[1] A\nSCIM is Enterprise only."]
    assert not check_answer("SCIM is Enterprise only.", ex, "q")[0]["ok"]
    assert not check_answer("SCIM is Enterprise only [3].", ex, "q")[0]["ok"]
    assert check_answer(REFUSAL, ex, "q")[0]["ok"]


def test_unknown_error_code_is_refused_before_any_model_call(svc):
    r = svc.ask("What does error KB-777 mean?")
    assert r["refused"] and r["gate"]["passed"] is False and r["attempts"] == 0
    assert "KB-777" in r["checks"][0]["detail"]


def test_known_error_code_retrieves_its_section_first(svc):
    r = svc.ask("my app says kb412")
    assert r["sources"][0]["section"] == "KB-412: clock out of sync"
    assert r["sources"][0]["exact_code"] is True


def test_out_of_scope_name_is_refused_without_a_model(svc):
    r = svc.ask("Does Kitebox integrate with Salesforce?")
    assert r["refused"] and "Salesforce" in r["gate"]["names_not_in_kb"]


def test_follow_up_keeps_the_subject(svc):
    hist = [{"role": "user", "content": "How much is the Team plan per user on annual billing?"},
            {"role": "assistant", "content": "Team is $10 per user per month on annual billing."}]
    fixed = svc.ask("And Business?", hist, "fixed")
    naive = svc.ask("And Business?", hist, "naive")
    assert "$20" in fixed["answer"]
    assert "Price per user" in [s["section"] for s in fixed["sources"]]
    assert "$20" not in naive["answer"]


def test_api_validates_input():
    from fastapi.testclient import TestClient

    from api.index import app

    c = TestClient(app)
    assert c.post("/api/chat", json={"message": "", "pipeline": "fixed"}).status_code == 422
    assert c.post("/api/chat", json={"message": "hi", "pipeline": "other"}).status_code == 422
    assert c.get("/api/kb/..%2F.env").status_code == 404
    r = c.post("/api/chat", json={"message": "Is there a Linux desktop app?"})
    assert r.status_code == 200 and "Linux" in r.json()["answer"]
