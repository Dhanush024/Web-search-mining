"""
Module 1: Document Loading
==========================
Responsible for loading text documents from the filesystem, directories,
or in-memory representations into structured Document objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class Document:
    """
    Represents an indexed document in the Information Retrieval system.

    Attributes:
        doc_id: Unique identifier for the document (integer or string).
        title: Human-readable title or filename.
        content: Raw text content of the document.
        file_path: Optional path to the source file on disk.
        metadata: Optional dictionary holding extra properties (e.g. file size, timestamp).
    """
    doc_id: int
    title: str
    content: str
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snippet(self, max_chars: int = 180) -> str:
        """Returns a shortened preview of the document text."""
        clean = " ".join(self.content.split())
        if len(clean) <= max_chars:
            return clean
        return clean[:max_chars].rstrip() + "..."

    def __repr__(self) -> str:
        return f"Document(id={self.doc_id}, title='{self.title}', len={len(self.content)})"


class DocumentLoader:
    """
    Loads documents from files, directories, or dictionaries with encoding fallbacks.
    """

    SUPPORTED_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")

    def __init__(self, default_extensions: Sequence[str] = (".txt", ".md", ".text")):
        self.default_extensions = tuple(ext.lower() for ext in default_extensions)

    def load_file(self, file_path: Union[str, Path], doc_id: int) -> Document:
        """
        Loads a single document from a file path.

        Args:
            file_path: Path to the text file.
            doc_id: Unique integer ID assigned to this document.

        Returns:
            Document instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be decoded with any supported encoding.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Document file not found: {path}")

        content: Optional[str] = None
        for enc in self.SUPPORTED_ENCODINGS:
            try:
                content = path.read_text(encoding=enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            raise ValueError(
                f"Could not decode file {path.name} with encodings: {self.SUPPORTED_ENCODINGS}"
            )

        # Extract title from the first non-empty line or fallback to file stem
        first_line = ""
        for line in content.splitlines():
            line_str = line.strip("# \t\r\n")
            if line_str:
                first_line = line_str
                break
        title = first_line if (first_line and len(first_line) < 80) else path.stem.replace("_", " ").title()

        metadata = {
            "file_size": path.stat().st_size,
            "filename": path.name,
            "modified_time": path.stat().st_mtime,
        }

        return Document(
            doc_id=doc_id,
            title=title,
            content=content,
            file_path=str(path),
            metadata=metadata,
        )

    def load_directory(
        self,
        directory_path: Union[str, Path],
        extensions: Optional[Sequence[str]] = None,
        sort_by_name: bool = True,
    ) -> List[Document]:
        """
        Scans a directory and loads all matching text files into Document objects.

        Args:
            directory_path: Directory path containing documents.
            extensions: Optional tuple/list of extensions to filter (e.g. ['.txt']).
            sort_by_name: Whether to sort files alphabetically before assigning IDs.

        Returns:
            List of loaded Document objects.
        """
        dir_path = Path(directory_path).resolve()
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        valid_exts = tuple(ext.lower() for ext in (extensions or self.default_extensions))
        files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]

        if sort_by_name:
            files.sort(key=lambda p: p.name.lower())

        documents: List[Document] = []
        for idx, file_p in enumerate(files, start=1):
            doc = self.load_file(file_p, doc_id=idx)
            documents.append(doc)

        return documents

    def load_from_dict(self, docs_dict: Dict[Union[int, str], str]) -> List[Document]:
        """
        Loads documents from an in-memory dictionary where key is id/title and value is text.

        Args:
            docs_dict: Dictionary {id_or_title: content_text}

        Returns:
            List of Document objects.
        """
        documents: List[Document] = []
        for idx, (key, text) in enumerate(docs_dict.items(), start=1):
            doc_id = key if isinstance(key, int) else idx
            title = str(key) if not isinstance(key, int) else f"Document {key}"
            documents.append(
                Document(
                    doc_id=doc_id,
                    title=title,
                    content=text,
                    file_path=None,
                    metadata={"source": "in_memory"},
                )
            )
        return documents

