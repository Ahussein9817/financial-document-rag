import json

import pytest

import fetch_filings as ff


def test_fiscal_year_from_text():
    assert ff.fiscal_year_from_text("For the fiscal year ended December 31, 2024") == 2024
    assert ff.fiscal_year_from_text("FOR THE FISCAL YEAR  ENDED  December 31,2023") == 2023
    assert ff.fiscal_year_from_text("no cover page") is None


def test_pick_primary_document_ignores_exhibits_and_index():
    idx = {"directory": {"item": [
        {"name": "jpm-20241231.htm", "size": "9000000"},
        {"name": "jpm-20241231xex21.htm", "size": "9500000"},
        {"name": "0000019617-25-000270-index.htm", "size": "5000"},
        {"name": "a.xml", "size": "1"}]}}
    assert ff.pick_primary_document(idx) == "jpm-20241231.htm"
    with pytest.raises(ValueError):
        ff.pick_primary_document({"directory": {"item": []}})


def test_fetch_filing_writes_file_and_manifest_entry(tmp_path):
    index = json.dumps({"directory": {"item": [{"name": "jpm-20241231.htm", "size": "100"}]}}).encode()
    body = b"<html><body>For the fiscal year ended December 31, 2024 ...</body></html>"
    calls = []

    def fake_get(url, ua):
        calls.append((url, ua))
        return index if url.endswith("-index.json") else body

    e = ff.fetch_filing("JPM", "0000019617-25-000270", tmp_path, "Test test@example.com",
                        get=fake_get, sleep=lambda s: None)
    assert e["file"] == "JPM_10K_2024.htm" and e["fiscal_year_from_text"] == 2024
    assert (tmp_path / e["file"]).read_bytes() == body
    assert e["sha256"] == ff.sha256_bytes(body) and e["bytes"] == len(body)
    assert all(ua == "Test test@example.com" for _, ua in calls) and len(calls) == 2
    assert "/data/19617/000001961725000270/" in calls[0][0]
