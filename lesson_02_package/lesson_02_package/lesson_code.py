"""
Day 2 - Ingestion: Getting Real Data Into the System

Turns raw documents into a flat store of overlapping, uniformly-sized
passages. This is the passage store Day 4's retrieval stage will search.
"""

from dataclasses import dataclass
import json


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class Document:
    doc_id: str
    source: str
    raw_text: str


@dataclass
class Passage:
    passage_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int

    def to_dict(self) -> dict:
        return {
            "passage_id": self.passage_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


# ---------------------------------------------------------------------------
# Mock source documents (Day 2 works on realistic raw text, not a database -
# a real system would load these from files or a CMS; the chunking logic
# below is what matters, and it's identical either way).
# ---------------------------------------------------------------------------

MOCK_DOCUMENTS = [
    Document(
        doc_id="doc_refund",
        source="refund_policy.txt",
        raw_text=(
            "Annual plans can be refunded within 30 days of purchase, no "
            "questions asked. To request a refund, contact support with your "
            "order ID. Refunds are processed within 5 to 7 business days and "
            "returned to the original payment method. Monthly plans are "
            "non-refundable after the billing date has passed, but you can "
            "cancel at any time to stop future charges. Enterprise plans "
            "follow the terms in your signed contract, which supersede this "
            "policy. If you were charged in error, contact billing support "
            "directly rather than filing a standard refund request."
        ),
    ),
    Document(
        doc_id="doc_cancel",
        source="cancellation.txt",
        raw_text=(
            "You can cancel your subscription at any time from account "
            "settings. Cancellation takes effect at the end of your current "
            "billing period, and you keep access until then. There are no "
            "cancellation fees on any plan."
        ),
    ),
    Document(
        doc_id="doc_pricing",
        source="pricing.txt",
        raw_text=(
            "Pricing starts at $39 per month, or $279.30 per year on the "
            "annual plan, which is roughly 40% cheaper than paying monthly. "
            "Lifetime access is also available as a separate one-time plan."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Ingestor
# ---------------------------------------------------------------------------

class DocumentIngestor:
    def __init__(self, chunk_size: int = 220, overlap: int = 40):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_documents(self) -> list[Document]:
        """Loads raw documents. Today this returns the mock set above -
        swap this for real file/database reads without touching chunking."""
        return MOCK_DOCUMENTS

    def chunk_document(self, document: Document) -> list[Passage]:
        """Splits one document into overlapping, fixed-size passages."""
        text = document.raw_text
        passages: list[Passage] = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                passages.append(
                    Passage(
                        passage_id=f"{document.doc_id}_p{index}",
                        doc_id=document.doc_id,
                        text=chunk_text,
                        char_start=start,
                        char_end=end,
                    )
                )
                index += 1

            if end == len(text):
                break
            start = end - self.overlap

        return passages

    def ingest_all(self) -> list[Passage]:
        """Loads and chunks every document into one flat passage store."""
        all_passages: list[Passage] = []
        for document in self.load_documents():
            passages = self.chunk_document(document)
            print(f"Document '{document.source}' -> {len(passages)} passages")
            all_passages.extend(passages)
        return all_passages

    def save_passage_store(self, passages: list[Passage], path: str = "passages.json"):
        with open(path, "w") as f:
            json.dump([p.to_dict() for p in passages], f, indent=2)


if __name__ == "__main__":
    ingestor = DocumentIngestor()

    documents = ingestor.load_documents()
    print(f"Loaded {len(documents)} documents")

    all_passages = ingestor.ingest_all()
    print(f"Total passages ingested: {len(all_passages)}")

    ingestor.save_passage_store(all_passages)
    print("Passage store saved -> passages.json")
