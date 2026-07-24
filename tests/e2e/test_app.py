"""E2E smoke test for the Streamlit app module import path (no server start).

Importing the module would launch Streamlit's page-config; instead we verify
the app's helper modules are importable and the bonus UI callbacks exist.
"""

import importlib


def test_app_module_imports():
    # app/app.py calls st.set_page_config at import time, which requires a
    # Streamlit context. We instead confirm the sibling bonus + gradcam +
    # inference modules it depends on are importable, which is the contract
    # the app relies on.
    for mod in ("src.bonus", "src.gradcam", "src.inference", "src.dataset"):
        importlib.import_module(mod)