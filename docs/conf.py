project = "bidszarr"
copyright = "2026, bidszarr contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# autodoc pulls docstrings straight from the installed package -- no sys.path
# hack needed since `pip install -e .` already put it on the venv's path.
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False}

html_theme = "furo"
