"""
Unit tests for Module 7: Frontend GUI
"""

import tkinter as tk
import unittest
from pathlib import Path

from src.gui import SearchEngineGUI


class TestSearchEngineGUI(unittest.TestCase):

    def setUp(self):
        self.root = tk.Tk()
        # Withdraw window so it doesn't pop up and block headless/test runs
        self.root.withdraw()
        data_dir = str(Path(__file__).resolve().parent.parent / "data")
        self.app = SearchEngineGUI(self.root, default_corpus_dir=data_dir)

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_gui_components_exist(self):
        # 1. Search input box exists
        self.assertIsNotNone(self.app.search_entry)
        self.assertIsInstance(self.app.search_entry, tk.Entry)

        # 2. Search button exists
        self.assertIsNotNone(self.app.search_btn)

        # 3. Read-only result area exists and is set to disabled
        self.assertIsNotNone(self.app.result_text)
        self.assertEqual(str(self.app.result_text.cget("state")), "disabled")

    def test_gui_search_execution(self):
        self.app.search_var.set("retrieval AND index")
        self.app.mode_var.set("Boolean (AND/OR/NOT)")
        self.app.execute_search()

        # Result area should still be read-only (disabled) after search
        self.assertEqual(str(self.app.result_text.cget("state")), "disabled")

        # Verify content was inserted into result_text
        content = self.app.result_text.get("1.0", tk.END)
        self.assertIn("Found 3 matching document(s)", content)
        self.assertIn("Doc1", content)


if __name__ == "__main__":
    unittest.main()

