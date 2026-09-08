import os

project = "neurozarr"
copyright = "2026, neurozarr contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
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
if os.environ.get("NEUROZARR_DOCS_OFFLINE"):
    intersphinx_mapping = {}

# autodoc pulls docstrings straight from the installed package -- no sys.path
# hack needed since `pip install -e .` already put it on the venv's path.
# Turn types named in numpydoc Parameters/Returns sections into real links
# rather than plain text.
napoleon_preprocess_types = True

autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False}

# Without this, every autodoc'd method signature (Repo.create(), Subject.add_visit(), ...)
# becomes its own entry in the page's sidebar TOC -- unusable once a class has more than
# a handful of members. The API page's own "At a glance" tables are the intended navigation.
toc_object_entries = False

html_theme = "furo"
