#!/usr/bin/env python3
"""Launch the Path B drag-and-drop mastering UI in a local browser."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui.chain_app import app

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", inbrowser=True)
