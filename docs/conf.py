project = "neurozarr"
copyright = "2026, neurozarr contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

# Link types like numpy.ndarray and mne.io.RawArray to their own docs. Needs
# network access on the first build; without it Sphinx warns and carries on.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "mne": ("https://mne.tools/stable", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable", None),
}

# autodoc pulls docstrings straight from the installed package -- no sys.path
# hack needed since `pip install -e .` already put it on the venv's path.
# Turn types named in numpydoc Parameters/Returns sections into real links
# rather than plain text.
napoleon_preprocess_types = True

autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False}

html_theme = "furo"
