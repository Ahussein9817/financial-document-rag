import ingestion


class FakeDoc:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {}


def _patch_loader(monkeypatch, pages_per_pdf=2):
    class FakeLoader:
        def __init__(self, path): self.path = path  # noqa: E701

        def load(self):
            return [FakeDoc(f"page {i}") for i in range(pages_per_pdf)]

    monkeypatch.setattr(ingestion, "PyPDFLoader", FakeLoader)


def test_load_pdfs_adds_ticker_year_source_metadata(tmp_path, monkeypatch):
    (tmp_path / "JPM_10K_2024.pdf").write_bytes(b"")
    (tmp_path / "BAC_10K_2022.pdf").write_bytes(b"")
    _patch_loader(monkeypatch)
    docs = ingestion.load_pdfs(str(tmp_path))
    assert len(docs) == 4
    metas = {(d.metadata["ticker"], d.metadata["year"], d.metadata["source_file"]) for d in docs}
    assert metas == {("JPM", "2024", "JPM_10K_2024.pdf"), ("BAC", "2022", "BAC_10K_2022.pdf")}


def test_load_pdfs_badly_named_file_gets_no_metadata(tmp_path, monkeypatch):
    # documents current behaviour: such a file can never match a metadata filter
    (tmp_path / "annual_report.pdf").write_bytes(b"")
    _patch_loader(monkeypatch)
    docs = ingestion.load_pdfs(str(tmp_path))
    assert docs and all("ticker" not in d.metadata for d in docs)


def test_load_pdfs_empty_directory(tmp_path):
    assert ingestion.load_pdfs(str(tmp_path)) == []


def test_chunk_documents_respects_size_and_overlap_and_keeps_metadata():
    from langchain_core.documents import Document
    text = " ".join(f"word{i}" for i in range(400))
    doc = Document(page_content=text, metadata={"ticker": "C", "year": "2023"})
    chunks = ingestion.chunk_documents([doc], chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 3
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata["ticker"] == "C" and c.metadata["year"] == "2023" for c in chunks)
    assert all("start_index" in c.metadata for c in chunks)
    # overlap: consecutive chunks share some text
    assert set(chunks[0].page_content.split()[-3:]) & set(chunks[1].page_content.split())


def test_save_sample_chunks_truncates_content(tmp_path):
    import json

    from langchain_core.documents import Document
    out = tmp_path / "s" / "samples.json"
    ingestion.save_sample_chunks([Document(page_content="x" * 900, metadata={"a": 1})], str(out), n=1)
    rec = json.loads(out.read_text())[0]
    assert rec["length"] == 900 and rec["content"].endswith("...") and len(rec["content"]) == 503


def test_assign_chunk_ids_are_stable_and_page_based():
    from langchain_core.documents import Document
    docs = [Document(page_content=f"c{i}", metadata={"source_file": "A_10K_2024.htm", "page": p})
            for i, p in enumerate([0, 0, 1])]
    docs.append(Document(page_content="x", metadata={"source_file": "B.htm"}))   # no page
    ingestion.assign_chunk_ids(docs)
    assert [d.metadata["chunk_id"] for d in docs[:3]] == [
        "A_10K_2024.htm#p1#c0", "A_10K_2024.htm#p1#c1", "A_10K_2024.htm#p2#c0"]
    assert docs[0].metadata["passage_id"] == "A_10K_2024.htm#p1"
    assert len(docs[0].metadata["content_sha1"]) == 16
    assert "chunk_id" not in docs[3].metadata
    again = ingestion.assign_chunk_ids([Document(page_content="c0", metadata={"source_file": "A_10K_2024.htm", "page": 0})])
    assert again[0].metadata["content_sha1"] == docs[0].metadata["content_sha1"]


def test_chunk_documents_sets_ids_and_start_index():
    from langchain_core.documents import Document
    doc = Document(page_content=" ".join(f"w{i}" for i in range(300)),
                   metadata={"source_file": "A_10K_2024.htm", "page": 4})
    chunks = ingestion.chunk_documents([doc], chunk_size=200, chunk_overlap=20)
    assert [c.metadata["chunk_id"] for c in chunks][:2] == ["A_10K_2024.htm#p5#c0", "A_10K_2024.htm#p5#c1"]
    assert chunks[0].metadata["start_index"] == 0 and chunks[1].metadata["start_index"] > 0


def _manifest(tmp_path, html, **over):
    import hashlib
    import json
    (tmp_path / "JPM_10K_2024.htm").write_text(html)
    e = {"ticker": "JPM", "accession": "0000019617-25-000270", "file": "JPM_10K_2024.htm",
         "fiscal_year": 2024, "period_of_report": "2024-12-31", "fiscal_year_source": "ixbrl_dei",
         "sha256": hashlib.sha256(html.encode()).hexdigest()}
    e.update(over)
    (tmp_path / "manifest.json").write_text(json.dumps({"filings": [e]}))


def test_load_filings_uses_manifest_provenance_and_html_pages(tmp_path):
    _manifest(tmp_path, '<p>one</p><hr style="page-break-after:always"><p>two</p>')
    docs = ingestion.load_filings(str(tmp_path))
    assert [d.metadata["page"] for d in docs] == [0, 1] and docs[1].page_content == "two"
    m = docs[0].metadata
    assert (m["ticker"], m["year"], m["accession"], m["period_of_report"], m["fiscal_year_source"]) == (
        "JPM", "2024", "0000019617-25-000270", "2024-12-31", "ixbrl_dei")


def test_load_filings_refuses_unresolved_year_and_hash_mismatch(tmp_path):
    import pytest
    _manifest(tmp_path, "<p>x</p>", fiscal_year=None)
    with pytest.raises(ValueError, match="fiscal year not resolved"):
        ingestion.load_filings(str(tmp_path))
    _manifest(tmp_path, "<p>x</p>", sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        ingestion.load_filings(str(tmp_path))


def test_legacy_pdf_loader_flags_filename_derived_year(tmp_path, monkeypatch):
    (tmp_path / "JPM_10K_2024.pdf").write_bytes(b"")
    _patch_loader(monkeypatch, pages_per_pdf=1)
    docs = ingestion.load_pdfs(str(tmp_path))
    assert docs[0].metadata["fiscal_year_source"] == "filename"
