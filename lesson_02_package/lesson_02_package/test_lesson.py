"""Day 2 tests - verify ingestion and chunking behave correctly."""

import json
import os

from lesson_code import DocumentIngestor, Document


def test_ingest_all_produces_five_passages():
    ingestor = DocumentIngestor()
    passages = ingestor.ingest_all()
    assert len(passages) == 5


def test_passage_ids_are_unique():
    ingestor = DocumentIngestor()
    passages = ingestor.ingest_all()
    ids = [p.passage_id for p in passages]
    assert len(ids) == len(set(ids))


def test_chunking_respects_overlap():
    ingestor = DocumentIngestor(chunk_size=100, overlap=20)
    document = Document(
        doc_id="doc_test",
        source="test.txt",
        raw_text="A" * 250,
    )
    passages = ingestor.chunk_document(document)
    assert len(passages) > 1
    # consecutive passages should share overlapping character ranges
    first, second = passages[0], passages[1]
    assert second.char_start < first.char_end


def test_short_document_produces_single_passage():
    ingestor = DocumentIngestor(chunk_size=220, overlap=40)
    document = Document(
        doc_id="doc_short",
        source="short.txt",
        raw_text="This is a short document under the chunk size limit.",
    )
    passages = ingestor.chunk_document(document)
    assert len(passages) == 1
    assert passages[0].char_start == 0


def test_save_passage_store_writes_valid_json(tmp_path):
    ingestor = DocumentIngestor()
    passages = ingestor.ingest_all()
    out_path = tmp_path / "passages.json"
    ingestor.save_passage_store(passages, path=str(out_path))

    assert os.path.exists(out_path)
    with open(out_path) as f:
        data = json.load(f)
    assert len(data) == 5
    assert "passage_id" in data[0]
