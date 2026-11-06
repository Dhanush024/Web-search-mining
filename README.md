# Web Search and Mining (WSM) - Information Retrieval Engine

A modular Information Retrieval (IR) and Web Search system built from scratch in Python. This project fulfills all requirements for the **WSM Assignment**, featuring a 7-module architecture, zero mandatory external dependencies, and a modern GUI.

---

## 📁 Clean Project Structure

```text
Web_search_mining/
│
├── data/                           # Sample corpus for indexing and testing
│   ├── doc1.txt                    # Information Retrieval and Search Engines
│   ├── doc2.txt                    # Web Crawlers, PageRank, and Web Mining
│   ├── doc3.txt                    # Wildcard Queries & Permuterm Indexing
│   ├── doc4.txt                    # NLP Preprocessing, Case Folding & Stemming
│   ├── doc5.txt                    # Vector Space Models & TF-IDF Cosine Ranking
│   └── doc6.txt                    # Boolean Retrieval & Evaluation Metrics
│
├── src/                            # Core Source Modules
│   ├── __init__.py                 # Unified package exports
│   ├── document_loader.py          # [Module 1] Document Loading & Corpus Management
│   ├── tokenizer.py                # [Module 2] Lexical Tokenization & Offset Tracking
│   ├── normalizer.py               # [Module 3] Case Folding, Accents, Stopwords & Porter Stemmer
│   ├── inverted_index.py           # [Module 4] Positional Inverted Index & JSON Serialization
│   ├── query_processor.py          # [Module 5] Boolean, Phrasal, and TF-IDF Ranked Querying
│   ├── wildcard_processor.py       # [Module 6] Permuterm Index, K-Gram & Wildcard Expansion
│   └── gui.py                      # [Module 7] Modern Tkinter Desktop GUI
│
├── tests/                          # Automated Unit Test Suite (30 Tests)
│   ├── __init__.py
│   ├── test_document_loader.py     # Tests file reading, encoding fallbacks, snippets
│   ├── test_tokenizer.py           # Tests token extraction and positional offsets
│   ├── test_normalizer.py          # Tests case folding, diacritics, and Porter stemmer
│   ├── test_inverted_index.py      # Tests postings lists, vocabulary, and JSON serialization
│   ├── test_wildcard_processor.py  # Tests Permuterm rotations, leading/trailing/middle wildcards
│   ├── test_query_processor.py     # Tests Boolean, exact phrase, and ranked TF-IDF searches
│   └── test_gui.py                 # Tests GUI components and read-only state integrity
│
├── main.py                         # Application entrypoint (GUI & CLI modes)
├── requirements.txt                # Standard library compatible (optional pytest)
└── README.md                       # Comprehensive documentation
```

---

## 🧩 The 7 Modules Explained

### 1. Document Loading (`src/document_loader.py`)

- **`Document` Dataclass**: Encapsulates `doc_id`, `title`, `content`, `file_path`, and metadata.
- **`DocumentLoader`**:
  - `load_directory()`: Traverses directories and ingests files with automatic encoding detection (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`).
  - `load_file()`: Loads individual text files, deriving intelligent titles from content.
  - `load_from_dict()`: In-memory document loading for fast testing.

### 2. Tokenization (`src/tokenizer.py`)

- **`Token` Dataclass**: Captures word string, 0-indexed positional offset within document stream, and character start/end spans.
- **`Tokenizer`**: Splits text into terms while handling punctuation, numbers, hyphens, and apostrophes. Supports both simple string extraction (`tokenize()`) and positional tracking (`tokenize_with_positions()`).

### 3. Text Normalization / Case Folding (`src/normalizer.py`)

- **Case Folding**: Converts all characters to lowercase for case-insensitive matching.
- **Diacritic / Accent Stripping**: Normalizes Unicode forms (e.g., `résumé` → `resume`).
- **Stopwords Removal**: Built-in set of standard English stopwords with toggle capability.
- **Built-in Porter Stemmer**: Complete pure-Python implementation of the Martin Porter (1980) algorithm. Stems morphological variants (e.g., `crawlers` → `crawler`, `retrieving` → `retriev`) without external libraries like NLTK.

### 4. Inverted Index Construction (`src/inverted_index.py`)

- **Positional Postings Representation**:

  $$\text{term} \longrightarrow \left[ (\text{doc\_id}_1, \text{tf}_1, [p_1, p_2, \dots]), (\text{doc\_id}_2, \text{tf}_2, [p_1, \dots]) \right]$$

- **Metrics Tracked**: Document Frequency ($\text{DF}_t$), Term Frequency ($\text{TF}_{t,d}$), Document Lengths, and Vocabulary.
- **Serialization**: Full JSON save and restore support via `save_to_json()` and `load_from_json()`.

### 5. Query Processing (`src/query_processor.py`)

Supports multiple query modalities:

- **Boolean Queries**: `AND`, `OR`, `NOT` with Shunting-Yard postfix conversion, operator precedence ($\text{NOT} > \text{AND} > \text{OR}$), parentheses, and implicit AND insertion (e.g., `retrieval NOT mining`).
- **Exact Phrase Queries**: Enclosed in quotes (e.g., `"inverted index"`). Checks positional adjacency $p_{i+1} = p_i + 1$.
- **Ranked Retrieval (Vector Space Model)**:
  - **TF-IDF weighting**:

    $$w_{t,d} = (1 + \log_{10} \text{tf}_{t,d}) \times \log_{10}\left(1 + \frac{N}{\text{df}_t}\right)$$

  - Cosine length normalization for fair document comparison.
- **Snippet Generator**: Generates context windows around matched query terms with highlighting tags.

### 6. Wildcard Processing (`src/wildcard_processor.py`)

- **Permuterm Index**:
  - Appends sentinel `$` (e.g., `search$`) and stores all cyclic rotations (`search$`, `earch$s`, `arch$se`, `$search`).
  - Wildcard rotation rules:
    - Trailing `X*` → prefix lookup `$X`
    - Leading `*X` → prefix lookup `X$`
    - Middle `X*Y` → prefix lookup `Y$X`

- **K-Gram Index**: Character bi-grams and tri-grams to support multi-wildcard patterns and substring queries (e.g., `*ind*`).
- **Regex Post-filtering**: Accurate verification for wildcard queries containing `*` or `?`.

### 7. Frontend GUI (`src/gui.py`)

A modern desktop GUI built using Python's native `tkinter` and `ttk`:

- **Search Input Box**: Clean `tk.Entry` with placeholder instructions and `<Return>` key execution.
- **Search Button**: Prominent "Search 🔍" action button with visual feedback.
- **Read-Only Result Area**: Scrollable `tk.Text` widget set to `state="disabled"` to guarantee that users cannot alter the displayed results. Features syntax-colored badges for Document Titles, Scores, Matched Terms, and Snippets.
- **Query Mode Selector**: Auto-detect, Ranked (TF-IDF), Boolean (AND/OR/NOT), Phrase ("..."), or Wildcard (*, ?).
- **Corpus Directory Browser**: Interactive folder selection to index any custom corpus of text files.
- **Quick Example Queries**: One-click test chips for immediate testing.
- **Live Status Bar**: Displays loaded document count, vocabulary size, and query search latency (in ms).

---

## 🚀 How to Run

### Requirements

- Python 3.8+ (standard library only; no `pip install` required for running).

### 1. Launch Frontend GUI (Default)

```bash
python main.py
```

Or specify a custom document directory:

```bash
python main.py --data ./data
```

### 2. Run via Command Line Interface (CLI Mode)

Useful for quick testing or headless evaluation:

```bash
# Boolean query:
python main.py --cli --query "retrieval AND index NOT mining"

# Wildcard query:
python main.py --cli --query "pro*"

# Leading wildcard:
python main.py --cli --query "*mining"

# Exact phrase query:
python main.py --cli --mode phrase --query "inverted index"

# Ranked TF-IDF query:
python main.py --cli --query "vector space models tf-idf"
```

### 3. Run Automated Unit Tests

A comprehensive test suite of 30 unit tests covers every module:

```bash
python -m unittest discover -s tests
```

---

## 🧪 Example Test Queries & Outputs

| Query | Mode | Expected Behavior / Matches |
| :--- | :--- | :--- |
| `pro*` | Wildcard | Expands to `problem`, `processing`, `processor`, `proportion` |
| `*mining` | Wildcard | Resolves leading wildcard to `mining`, `datamining` |
| `"web mining"` | Phrase | Finds documents where `web` is immediately followed by `mining` |
| `retrieval AND index` | Boolean | Intersects posting lists; matches Doc 1, Doc 3, Doc 6 |
| `retrieval NOT mining` | Boolean | Set difference; excludes documents containing `mining` |
| `vector space models` | Ranked | Computes TF-IDF cosine similarity scores; ranks Doc 5 highest |