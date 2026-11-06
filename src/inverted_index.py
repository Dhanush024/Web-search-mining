"""
Module 4: Inverted Index Construction
=====================================
Builds and manages a positional inverted index supporting:
- Document indexing with term frequency (TF) and word positions
- Vocabulary tracking
- Document frequency (DF) calculations
- Statistics computation (corpus size, vocabulary size, avg length)
- Serialization to / deserialization from JSON
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.document_loader import Document
from src.normalizer import TextNormalizer
from src.tokenizer import Tokenizer


@dataclass
class Posting:
    """
    Represents an entry in a term's postings list.

    Attributes:
        doc_id: The identifier of the document containing the term.
        tf: Term frequency (number of occurrences in this document).
        positions: Sorted list of 0-based word positions where the term occurs.
    """
    doc_id: int
    tf: int = 0
    positions: List[int] = field(default_factory=list)

    def add_position(self, pos: int) -> None:
        self.positions.append(pos)
        self.tf = len(self.positions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "tf": self.tf,
            "positions": self.positions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Posting:
        return cls(
            doc_id=data["doc_id"],
            tf=data.get("tf", len(data.get("positions", []))),
            positions=data.get("positions", []),
        )


class InvertedIndex:
    """
    Positional Inverted Index structure.
    Maps term -> List of Posting objects.
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        normalizer: Optional[TextNormalizer] = None,
    ):
        self.tokenizer = tokenizer or Tokenizer()
        self.normalizer = normalizer or TextNormalizer()

        # Core index structure: term -> {doc_id: Posting}
        self.index: Dict[str, Dict[int, Posting]] = {}

        # Document registry: doc_id -> Document
        self.documents: Dict[int, Document] = {}

        # Document length (number of tokens) for normalization: doc_id -> token_count
        self.doc_lengths: Dict[int, int] = {}

    def clear(self) -> None:
        """Resets the index and all document records."""
        self.index.clear()
        self.documents.clear()
        self.doc_lengths.clear()

    @property
    def total_documents(self) -> int:
        """Total number of indexed documents (N)."""
        return len(self.documents)

    @property
    def vocabulary_size(self) -> int:
        """Total number of unique terms in the index."""
        return len(self.index)

    def get_vocabulary(self) -> Set[str]:
        """Returns the set of unique indexed terms."""
        return set(self.index.keys())

    def add_document(self, document: Document) -> None:
        """
        Tokenizes, normalizes, and indexes a single Document object.

        Args:
            document: Document instance to be indexed.
        """
        doc_id = document.doc_id
        self.documents[doc_id] = document

        # Extract tokens with positions
        tokens_with_pos = self.tokenizer.tokenize_with_positions(document.content)
        total_tokens = 0

        for tok in tokens_with_pos:
            norm_term = self.normalizer.normalize(tok.text)
            if not norm_term:
                continue

            total_tokens += 1

            if norm_term not in self.index:
                self.index[norm_term] = {}

            postings_map = self.index[norm_term]
            if doc_id not in postings_map:
                postings_map[doc_id] = Posting(doc_id=doc_id)

            postings_map[doc_id].add_position(tok.position)

        self.doc_lengths[doc_id] = total_tokens

    def build(self, documents: Sequence[Document]) -> None:
        """
        Builds the inverted index from a sequence of documents.

        Args:
            documents: Sequence of Document objects.
        """
        self.clear()
        for doc in documents:
            self.add_document(doc)

    def get_postings(self, term: str) -> List[Posting]:
        """
        Returns the postings list for a given term, sorted by doc_id.

        Args:
            term: Term string (will be normalized).

        Returns:
            List of Posting objects.
        """
        norm = self.normalizer.normalize(term)
        if not norm or norm not in self.index:
            return []
        # Return sorted by doc_id
        return sorted(self.index[norm].values(), key=lambda p: p.doc_id)

    def get_doc_frequency(self, term: str) -> int:
        """
        Returns document frequency (DF_t): number of documents containing the term.
        """
        norm = self.normalizer.normalize(term)
        if not norm or norm not in self.index:
            return 0
        return len(self.index[norm])

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """
        Returns term frequency (TF_{t,d}): number of times term appears in doc_id.
        """
        norm = self.normalizer.normalize(term)
        if not norm or norm not in self.index:
            return 0
        posting = self.index[norm].get(doc_id)
        return posting.tf if posting else 0

    def get_document_ids(self, term: str) -> List[int]:
        """Returns sorted list of doc_ids that contain the term."""
        return [p.doc_id for p in self.get_postings(term)]

    def get_all_doc_ids(self) -> List[int]:
        """Returns sorted list of all indexed doc_ids."""
        return sorted(self.documents.keys())

    def get_document(self, doc_id: int) -> Optional[Document]:
        """Retrieves Document by doc_id."""
        return self.documents.get(doc_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Returns summary statistics about the constructed index."""
        total_postings = sum(len(postings_map) for postings_map in self.index.values())
        avg_doc_len = (
            sum(self.doc_lengths.values()) / self.total_documents
            if self.total_documents > 0
            else 0.0
        )
        return {
            "total_documents": self.total_documents,
            "vocabulary_size": self.vocabulary_size,
            "total_postings": total_postings,
            "average_document_length": round(avg_doc_len, 2),
        }

    def save_to_json(self, file_path: Union[str, Path]) -> None:
        """Serializes the index and document metadata to a JSON file."""
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable_index = {
            term: [p.to_dict() for p in sorted(postings_map.values(), key=lambda x: x.doc_id)]
            for term, postings_map in self.index.items()
        }

        serializable_docs = {
            str(doc_id): {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "content": doc.content,
                "file_path": doc.file_path,
                "metadata": doc.metadata,
            }
            for doc_id, doc in self.documents.items()
        }

        payload = {
            "index": serializable_index,
            "documents": serializable_docs,
            "doc_lengths": {str(k): v for k, v in self.doc_lengths.items()},
        }

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_from_json(self, file_path: Union[str, Path]) -> None:
        """Deserializes the index and document metadata from a JSON file."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Index file not found: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        self.clear()

        for doc_id_str, d in payload.get("documents", {}).items():
            doc_id = int(doc_id_str)
            self.documents[doc_id] = Document(
                doc_id=doc_id,
                title=d["title"],
                content=d["content"],
                file_path=d.get("file_path"),
                metadata=d.get("metadata", {}),
            )

        for doc_id_str, length in payload.get("doc_lengths", {}).items():
            self.doc_lengths[int(doc_id_str)] = length

        for term, postings_list in payload.get("index", {}).items():
            self.index[term] = {
                p_dict["doc_id"]: Posting.from_dict(p_dict)
                for p_dict in postings_list
            }

