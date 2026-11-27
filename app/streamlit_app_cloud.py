
import sys
import runpy
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load the cloud-specific API clients first.
import config.clients_streamlit as cloud_clients

# Route existing agent imports to the cloud clients.
# The original config/clients.py remains unchanged.
sys.modules["config.clients"] = cloud_clients

# Launch the original application.
runpy.run_path(
    str(ROOT_DIR / "app" / "streamlit_app.py"),
    run_name="__main__",
)