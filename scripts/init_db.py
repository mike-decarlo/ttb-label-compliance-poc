"""One-time setup: build the SQLite rules database.

Run with: python scripts/init_db.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.validation import build_rules_db

if __name__ == "__main__":
    build_rules_db()
    print("rules/label_rules.db created.")