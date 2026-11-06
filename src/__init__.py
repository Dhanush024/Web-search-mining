"""
Web Search and Mining (WSM) Information Retrieval Package
=========================================================

Clean modular architecture implementing:
1. Document Loading (document_loader.py)
2. Tokenization (tokenizer.py)
3. Text Normalization / Case Folding (normalizer.py)
4. Inverted Index Construction (inverted_index.py)
5. Query Processing (query_processor.py)
6. Wildcard Processing (wildcard_processor.py)
7. Frontend GUI (gui.py)
"""

from src.document_loader import Document, DocumentLoader
from src.tokenizer import Token, Tokenizer
from src.normalizer import PorterStemmer, TextNormalizer
from src.inverted_index import InvertedIndex, Posting
from src.wildcard_processor import KGramIndex, PermutermIndex, WildcardProcessor
from src.query_processor import QueryProcessor, SearchResult
from src.gui import SearchEngineGUI, launch_gui

__all__ = [
    "Document",
    "DocumentLoader",
    "Token",
    "Tokenizer",
    "PorterStemmer",
    "TextNormalizer",
    "Posting",
    "InvertedIndex",
    "PermutermIndex",
    "KGramIndex",
    "WildcardProcessor",
    "QueryProcessor",
    "SearchResult",
    "SearchEngineGUI",
    "launch_gui",
]

