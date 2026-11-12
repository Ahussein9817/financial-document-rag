import json

import pytest

import fetch_filings as ff


def test_pick_primary_document_ignores_exhibits_and_index():
    idx = {"directory": {"item": [
        {"name": "jpm-20241231.htm", "size": "9000000"},
        {"name": "jpm-20241231xex21.htm", "size": "9500000"},
        {"name": "0000019617-25-000270-index.htm", "size": "5000"},
        {"name": "a.xml", "size": "1"}]}}
    assert ff.pick_primary_document(idx) == "jpm-20241231.htm"
    with pytest.raises(ValueError):
        ff.pick_primary_document({"directory": {"item": []}})


def _index():
    return json.dumps({"directory": {"item": [{"name": "jpm-20241231.htm", "size": "100"}]}}).encode()


def test_pick_primary_document_real_listing_shape():
    # shape of an actual EDGAR listing: exhibits, XBRL viewer pages, empty sizes, a zip
    idx = {"directory": {"item": [
        {"name": "0000019617-25-000270-index.html", "size": ""},
        {"name": "corp10k2024exhibit21.htm", "size": "11883"},
        {"name": "R11.htm", "size": "1051304"},
        {"name": "jpm-20241231.htm", "size": "12849180"},
        {"name": "0000019617-25-000270-xbrl.zip", "size": "5764429"}]}}
    assert ff.pick_primary_document(idx) == "jpm-20241231.htm"


def test_index_url_is_the_json_directory_listing():
    assert ff.index_url("0000019617", "0000019617-25-000270").endswith("/000001961725000270/index.json")


def test_fetch_filing_reads_year_from_filing_not_filing_date(tmp_path):
    # filed in 2025 but the filing itself says fiscal year 2024
    body = (b'<html><ix:nonNumeric name="dei:DocumentType">10-K</ix:nonNumeric>'
            b'<ix:nonNumeric name="dei:DocumentPeriodEndDate">December 31, 2024</ix:nonNumeric>'
            b'<ix:nonNumeric name="dei:DocumentFiscalYearFocus">2024</ix:nonNumeric></html>')
    calls = []

    def fake_get(url, ua):
        calls.append((url, ua))
        return _index() if url.endswith("index.json") else body

    e = ff.fetch_filing("JPM", "0000019617-25-000270", tmp_path, "Test test@example.com",
                        get=fake_get, sleep=lambda s: None)
    assert e["file"] == "JPM_10K_2024.htm" and e["fiscal_year"] == 2024
    assert e["accession"] == "0000019617-25-000270" and e["period_of_report"] == "2024-12-31"
    assert e["form_type"] == "10-K" and e["fiscal_year_source"] == "ixbrl_dei"
    assert (tmp_path / e["file"]).read_bytes() == body
    assert e["sha256"] == ff.sha256_bytes(body) and e["bytes"] == len(body)
    assert all(ua == "Test test@example.com" for _, ua in calls) and len(calls) == 2
    assert "/data/19617/000001961725000270/" in calls[0][0]


def test_unresolvable_fiscal_year_is_flagged_not_guessed(tmp_path):
    def fake_get(url, ua):
        return _index() if url.endswith("index.json") else b"<html>no metadata here</html>"

    e = ff.fetch_filing("C", "0000831001-25-000067", tmp_path, "T t@e.com", get=fake_get, sleep=lambda s: None)
    assert e["fiscal_year"] is None and e["file"] == "C_10K_UNRESOLVED_0000831001-25-000067.htm"


def test_requests_are_rate_limited(tmp_path):
    sleeps = []

    def fake_get(url, ua):
        return _index() if url.endswith("index.json") else b"<html></html>"

    ff.fetch_filing("JPM", "0000019617-25-000270", tmp_path, "T t@e.com", get=fake_get, sleep=sleeps.append)
    assert sleeps == [ff.MIN_INTERVAL_S, ff.MIN_INTERVAL_S] and ff.MIN_INTERVAL_S >= 0.1   # <= 10 req/s


def test_reparse_manifest_updates_metadata_without_network(tmp_path):
    body = b'<ix:nonNumeric name="dei:DocumentPeriodEndDate">December 31, 2023</ix:nonNumeric>'
    (tmp_path / "X_10K_2023.htm").write_bytes(body)
    (tmp_path / "manifest.json").write_text(json.dumps({"filings": [
        {"file": "X_10K_2023.htm", "accession": "A", "fiscal_year": None, "period_of_report": None}]}))
    out = ff.reparse_manifest(tmp_path)
    assert out[0]["period_of_report"] == "2023-12-31" and out[0]["fiscal_year"] == 2023
    assert out[0]["accession"] == "A" and out[0]["locator_kind"] == "text_block"
