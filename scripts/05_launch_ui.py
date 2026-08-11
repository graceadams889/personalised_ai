#!/usr/bin/env python3
"""Step 5: Launch the Gradio mastering UI."""
import sys
sys.path.insert(0, ".")

from src.ui.app import create_app, load_config

if __name__ == "__main__":
    config = load_config()
    port = config.get("deployment", {}).get("gradio_port", 7860)
    app = create_app()
    app.launch(server_port=port)
