"""
Module 7: Frontend GUI
======================
Tkinter desktop user interface providing:
1. Search input box with placeholder support and Enter key binding
2. Search button with active visual feedback
3. Read-only result area (tk.Text with state='disabled' and rich tag formatting)
4. Query mode dropdown (Auto, Ranked TF-IDF, Boolean, Phrase, Wildcard)
5. Quick sample queries for one-click testing
6. Directory browser to load and index custom document collections
7. Real-time statistics bar (document count, vocabulary size, search latency)
"""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from src.document_loader import DocumentLoader
from src.inverted_index import InvertedIndex
from src.normalizer import TextNormalizer
from src.query_processor import QueryProcessor, SearchResult
from src.tokenizer import Tokenizer
from src.wildcard_processor import WildcardProcessor


class SearchEngineGUI:
    """
    Main application GUI for the Web Search and Mining Information Retrieval Engine.
    """

    PRIMARY_COLOR = "#1e293b"       # Slate 800
    ACCENT_COLOR = "#2563eb"        # Royal Blue
    ACCENT_HOVER = "#1d4ed8"        # Darker Blue
    BG_LIGHT = "#f8fafc"            # Slate 50
    TEXT_MAIN = "#0f172a"           # Slate 900
    TEXT_MUTED = "#64748b"          # Slate 500
    BORDER_COLOR = "#cbd5e1"        # Slate 300
    MATCH_BG = "#fef08a"            # Yellow highlight
    MATCH_FG = "#854d0e"

    def __init__(self, root: tk.Tk, default_corpus_dir: Optional[str] = None):
        self.root = root
        self.root.title("WSM Search Engine - Information Retrieval System")
        self.root.geometry("980x740")
        self.root.minsize(780, 560)
        self.root.configure(bg=self.BG_LIGHT)

        # Initialize IR Engine Components
        self.loader = DocumentLoader()
        self.tokenizer = Tokenizer()
        self.normalizer = TextNormalizer(case_fold=True, strip_accents=True)
        self.index = InvertedIndex(tokenizer=self.tokenizer, normalizer=self.normalizer)
        self.wildcard_processor = WildcardProcessor()
        self.query_processor = QueryProcessor(
            index=self.index,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
            wildcard_processor=self.wildcard_processor,
        )

        self.corpus_path = default_corpus_dir or str(Path(__file__).resolve().parent.parent / "data")

        # Setup GUI widgets and layout
        self._configure_styles()
        self._build_header()
        self._build_search_bar()
        self._build_quick_examples()
        self._build_result_area()
        self._build_status_bar()

        # Load initial corpus
        self._load_corpus(self.corpus_path)

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "TLabel",
            background=self.BG_LIGHT,
            foreground=self.TEXT_MAIN,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Header.TLabel",
            background=self.PRIMARY_COLOR,
            foreground="#ffffff",
            font=("Segoe UI", 15, "bold"),
        )
        style.configure(
            "SubHeader.TLabel",
            background=self.PRIMARY_COLOR,
            foreground="#94a3b8",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Action.TButton",
            font=("Segoe UI", 10, "bold"),
            background=self.ACCENT_COLOR,
            foreground="#ffffff",
            borderwidth=0,
            padding=(14, 7),
        )
        style.map(
            "Action.TButton",
            background=[("active", self.ACCENT_HOVER)],
        )
        style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 9),
            padding=(10, 4),
        )
        style.configure(
            "TCombobox",
            font=("Segoe UI", 10),
            padding=5,
        )

    def _build_header(self) -> None:
        header_frame = tk.Frame(self.root, bg=self.PRIMARY_COLOR, padx=20, pady=14)
        header_frame.pack(fill="x", side="top")

        left_box = tk.Frame(header_frame, bg=self.PRIMARY_COLOR)
        left_box.pack(side="left")

        title_lbl = ttk.Label(
            left_box,
            text="🔍 Web Search & Mining Engine",
            style="Header.TLabel",
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = ttk.Label(
            left_box,
            text="Positional Inverted Index • Permuterm Wildcards • Boolean & Ranked Retrieval",
            style="SubHeader.TLabel",
        )
        subtitle_lbl.pack(anchor="w", pady=(2, 0))

        right_box = tk.Frame(header_frame, bg=self.PRIMARY_COLOR)
        right_box.pack(side="right")

        btn_load = ttk.Button(
            right_box,
            text="📁 Change Corpus Directory",
            style="Secondary.TButton",
            command=self._choose_directory,
        )
        btn_load.pack(side="right", padx=5)

    def _build_search_bar(self) -> None:
        control_frame = tk.Frame(self.root, bg=self.BG_LIGHT, padx=20, pady=12)
        control_frame.pack(fill="x")

        # Search Box Container
        input_container = tk.Frame(control_frame, bg=self.BG_LIGHT)
        input_container.pack(fill="x", pady=2)

        # Search Input Box (Entry)
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(
            input_container,
            textvariable=self.search_var,
            font=("Segoe UI", 12),
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightcolor=self.ACCENT_COLOR,
            highlightbackground=self.BORDER_COLOR,
        )
        self.search_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        self.search_entry.bind("<Return>", lambda event: self.execute_search())

        # Query Mode Dropdown
        self.mode_var = tk.StringVar(value="Auto Detect")
        mode_options = [
            "Auto Detect",
            "Ranked (TF-IDF)",
            "Boolean (AND/OR/NOT)",
            "Phrase (\"...\")",
            "Wildcard (*, ?)",
        ]
        self.mode_combo = ttk.Combobox(
            input_container,
            textvariable=self.mode_var,
            values=mode_options,
            state="readonly",
            width=18,
            style="TCombobox",
        )
        self.mode_combo.pack(side="left", padx=(0, 8), ipady=3)

        # Search Button
        self.search_btn = ttk.Button(
            input_container,
            text="Search 🔍",
            style="Action.TButton",
            command=self.execute_search,
        )
        self.search_btn.pack(side="left")

        # Focus input on launch
        self.search_entry.focus_set()

    def _build_quick_examples(self) -> None:
        examples_frame = tk.Frame(self.root, bg=self.BG_LIGHT, padx=20, pady=2)
        examples_frame.pack(fill="x")

        lbl = ttk.Label(
            examples_frame,
            text="Quick Examples:",
            font=("Segoe UI", 9, "bold"),
            foreground=self.TEXT_MUTED,
        )
        lbl.pack(side="left", padx=(0, 8))

        samples = [
            ("Wildcard: pro*", "pro*", "Wildcard (*, ?)"),
            ("Leading: *mining", "*mining", "Wildcard (*, ?)"),
            ("Boolean: retrieval AND index", "retrieval AND index", "Boolean (AND/OR/NOT)"),
            ("Phrase: \"web mining\"", "\"web mining\"", "Phrase (\"...\")"),
            ("Ranked: vector space model", "vector space model", "Ranked (TF-IDF)"),
        ]

        for label, q_text, mode in samples:
            btn = tk.Button(
                examples_frame,
                text=label,
                font=("Segoe UI", 8),
                bg="#e2e8f0",
                fg=self.TEXT_MAIN,
                relief="flat",
                padx=6,
                pady=2,
                cursor="hand2",
                command=lambda qt=q_text, m=mode: self._run_sample(qt, m),
            )
            btn.pack(side="left", padx=3)

    def _build_result_area(self) -> None:
        result_frame = tk.Frame(self.root, bg=self.BG_LIGHT, padx=20, pady=8)
        result_frame.pack(fill="both", expand=True)

        lbl = ttk.Label(
            result_frame,
            text="Search Results (Read-Only Result Area):",
            font=("Segoe UI", 10, "bold"),
            foreground=self.TEXT_MAIN,
        )
        lbl.pack(anchor="w", pady=(0, 4))

        text_container = tk.Frame(result_frame, bg="#ffffff", bd=1, relief="solid")
        text_container.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(text_container, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        # Read-Only Result Area
        self.result_text = tk.Text(
            text_container,
            wrap="word",
            font=("Consolas", 10),
            bg="#ffffff",
            fg=self.TEXT_MAIN,
            padx=14,
            pady=12,
            yscrollcommand=scrollbar.set,
            relief="flat",
            state="disabled",  # READ-ONLY ENFORCED HERE
        )
        self.result_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.result_text.yview)

        # Configure formatting tags
        self.result_text.tag_configure("title", font=("Segoe UI", 12, "bold"), foreground="#1e3a8a")
        self.result_text.tag_configure("doc_id", font=("Segoe UI", 9, "bold"), foreground="#0369a1")
        self.result_text.tag_configure("score", font=("Segoe UI", 9, "bold"), foreground="#047857")
        self.result_text.tag_configure("meta", font=("Consolas", 9), foreground=self.TEXT_MUTED)
        self.result_text.tag_configure("snippet", font=("Segoe UI", 10), foreground="#334155")
        self.result_text.tag_configure(
            "highlight",
            font=("Segoe UI", 10, "bold"),
            background=self.MATCH_BG,
            foreground=self.MATCH_FG,
        )
        self.result_text.tag_configure("stats_banner", font=("Segoe UI", 10, "italic"), foreground="#475569")
        self.result_text.tag_configure("info_box", font=("Segoe UI", 10), foreground="#0f172a")
        self.result_text.tag_configure("divider", font=("Consolas", 8), foreground="#cbd5e1")

    def _build_status_bar(self) -> None:
        self.status_frame = tk.Frame(self.root, bg="#f1f5f9", padx=15, pady=6, bd=1, relief="sunken")
        self.status_frame.pack(fill="x", side="bottom")

        self.status_lbl = ttk.Label(
            self.status_frame,
            text="Ready. Corpus not loaded.",
            font=("Segoe UI", 9),
            foreground=self.TEXT_MUTED,
            background="#f1f5f9",
        )
        self.status_lbl.pack(side="left")

        self.timing_lbl = ttk.Label(
            self.status_frame,
            text="",
            font=("Segoe UI", 9, "bold"),
            foreground=self.ACCENT_COLOR,
            background="#f1f5f9",
        )
        self.timing_lbl.pack(side="right")

    # =========================================================================
    # Event Handlers & Core Actions
    # =========================================================================

    def _run_sample(self, query_text: str, mode: str) -> None:
        self.search_var.set(query_text)
        self.mode_var.set(mode)
        self.execute_search()

    def _choose_directory(self) -> None:
        selected = filedialog.askdirectory(
            title="Select Corpus Directory with Documents (.txt, .md)",
            initialdir=self.corpus_path,
        )
        if selected:
            self._load_corpus(selected)

    def _load_corpus(self, directory: str) -> None:
        dir_path = Path(directory).resolve()
        if not dir_path.is_dir():
            self._set_result_text(
                f"Directory not found: {dir_path}\nPlease select a valid directory containing text files."
            )
            return

        t0 = time.perf_counter()
        try:
            docs = self.loader.load_directory(dir_path)
            if not docs:
                self._set_result_text(
                    f"No text documents found in: {dir_path}\nPlease ensure the folder contains .txt or .md files."
                )
                return

            self.index.build(docs)
            self.query_processor.sync_wildcard_index()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            self.corpus_path = str(dir_path)
            stats = self.index.get_statistics()
            status_text = (
                f"📁 Corpus: {dir_path.name} | "
                f"Docs: {stats['total_documents']} | "
                f"Vocabulary: {stats['vocabulary_size']} terms | "
                f"Index Time: {elapsed_ms:.1f}ms"
            )
            self.status_lbl.config(text=status_text)

            welcome_msg = (
                f"=== WSM Information Retrieval Engine Initialized ===\n\n"
                f"Corpus Directory : {dir_path}\n"
                f"Documents Loaded : {stats['total_documents']}\n"
                f"Unique Terms     : {stats['vocabulary_size']}\n"
                f"Total Postings   : {stats['total_postings']}\n"
                f"Avg Doc Length   : {stats['average_document_length']} terms\n\n"
                f"Try queries such as:\n"
                f"  • Wildcard  : pro*  or  *mining  or  craw*er\n"
                f"  • Boolean   : retrieval AND index NOT web\n"
                f"  • Phrase    : \"information retrieval\"\n"
                f"  • Ranked    : vector space models tf-idf ranking\n"
            )
            self._set_result_text(welcome_msg)
        except Exception as ex:
            messagebox.showerror("Error Loading Corpus", str(ex))

    def execute_search(self) -> None:
        query = self.search_var.get().strip()
        if not query:
            self._set_result_text("Please enter a search query.")
            return

        mode_selection = self.mode_var.get().lower()
        mode_map = {
            "auto detect": "auto",
            "ranked (tf-idf)": "ranked",
            "boolean (and/or/not)": "boolean",
            'phrase ("...")': "phrase",
            "wildcard (*, ?)": "wildcard",
        }
        mode_key = mode_map.get(mode_selection, "auto")

        t0 = time.perf_counter()
        try:
            results, description = self.query_processor.search(query, mode=mode_key)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._render_results(query, results, description, elapsed_ms)
        except Exception as ex:
            self._set_result_text(f"Query Processing Error:\n{str(ex)}")

    def _set_result_text(self, text: str) -> None:
        """Helper to write raw text to read-only text area."""
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state="disabled")

    def _render_results(
        self,
        query: str,
        results: list[SearchResult],
        description: str,
        elapsed_ms: float,
    ) -> None:
        """
        Populates the read-only result area with structured, styled search results.
        """
        # Unlock read-only state for writing
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)

        self.timing_lbl.config(text=f"⏱ {elapsed_ms:.2f} ms")

        # Top summary banner
        self.result_text.insert(
            tk.END,
            f"Query: \"{query}\"  •  Mode: {description}\n",
            "stats_banner",
        )
        self.result_text.insert(
            tk.END,
            f"Found {len(results)} matching document(s) in {elapsed_ms:.2f} ms\n",
            "stats_banner",
        )
        self.result_text.insert(
            tk.END,
            "═" * 80 + "\n\n",
            "divider",
        )

        if not results:
            self.result_text.insert(
                tk.END,
                "No documents matched your query.\n\n"
                "Suggestions:\n"
                "• Check for spelling errors or try a wildcard query (e.g., search*).\n"
                "• For boolean queries, verify operators are uppercase: AND, OR, NOT.\n"
                "• Make sure terms exist in the loaded corpus.\n",
                "info_box",
            )
            self.result_text.config(state="disabled")
            return

        for rank, res in enumerate(results, start=1):
            # Result item header
            self.result_text.insert(tk.END, f"#{rank}  ", "score")
            self.result_text.insert(tk.END, f"{res.title}\n", "title")

            meta_str = f"    Doc ID: {res.doc_id}  |  Score: {res.score:.4f}"
            if res.file_path:
                meta_str += f"  |  File: {Path(res.file_path).name}"
            self.result_text.insert(tk.END, f"{meta_str}\n", "meta")

            if res.matched_terms:
                self.result_text.insert(tk.END, "    Matched Terms: ", "meta")
                self.result_text.insert(tk.END, f"{', '.join(res.matched_terms)}\n", "highlight")

            # Snippet excerpt
            self.result_text.insert(tk.END, "    Snippet: ", "meta")
            self.result_text.insert(tk.END, f"{res.snippet}\n", "snippet")

            self.result_text.insert(tk.END, "─" * 80 + "\n\n", "divider")

        # Re-lock to read-only state
        self.result_text.config(state="disabled")


def launch_gui(corpus_dir: Optional[str] = None) -> None:
    """Launches the Tkinter Search Engine GUI application."""
    root = tk.Tk()
    app = SearchEngineGUI(root, default_corpus_dir=corpus_dir)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()

