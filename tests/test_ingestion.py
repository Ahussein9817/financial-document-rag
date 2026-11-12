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
    assert all(c.metadata == {"ticker": "C", "year": "2023"} for c in chunks)
    # overlap: consecutive chunks share some text
    assert set(chunks[0].page_content.split()[-3:]) & set(chunks[1].page_content.split())


def test_save_sample_chunks_truncates_content(tmp_path):
    import json

    from langchain_core.documents import Document
    out = tmp_path / "s" / "samples.json"
    ingestion.save_sample_chunks([Document(page_content="x" * 900, metadata={"a": 1})], str(out), n=1)
    rec = json.loads(out.read_text())[0]
    assert rec["length"] == 900 and rec["content"].endswith("...") and len(rec["content"]) == 503
